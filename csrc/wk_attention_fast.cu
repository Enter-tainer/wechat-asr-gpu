/* wk_attention_fast.cu -- fused multi-head attention for the wtl ASR engine.
 *
 *   out[i][h*hd + lane] = sum_j softmax_j( q_i[h] . k_j[h] * 0.125 ) * v_j[h][lane]
 *
 * q/k/v/out are [t][d] row-major f32, heads = d/hd, hd = 64 (hard requirement here).
 *
 * Design (vs the "one block per (i,head)" version, which re-reads all of k and v
 * once per output row and is therefore L2-bandwidth bound):
 *   grid = (ceil(t/IT), heads) , block = 256 threads
 *   -> one block owns ONE head and IT consecutive output rows, so k/v for that head
 *      (t*64*4 bytes each) are streamed through shared memory once per IT rows
 *      instead of once per row: L2 traffic drops by a factor IT.
 *   QK  : scores[i][j] = dot64(q_sh[i], k_chunk[j]) * 0.125, k streamed in
 *         CH-row chunks into padded shared (stride 65 -> no bank conflicts).
 *   soft : per row warp-level reduce (no __syncthreads in the inner loop).
 *   PV  : accumulators live in registers (one per (i,lane) pair owned by the
 *         thread), so the j loop is fully pipelined / ILP=#pairs -- this is where
 *         the naive version loses (150-deep serial dependency chain).
 *
 * Build:
 *   /usr/local/cuda-13.0/bin/nvcc -O3 -arch=sm_120 -shared -Xcompiler -fPIC \
 *     -o libattn.so wk_attention_fast.cu -L/usr/local/cuda-13.0/lib64 -lcudart \
 *     -Xlinker -rpath -Xlinker /usr/local/cuda-13.0/lib64
 */
#include <cuda_runtime.h>
#include <cstdio>

#define HD 64
#define HDP (HD + 1)   /* padded shared stride */
#define NTH 256
#define MAXP 16        /* max (i,lane) pairs per thread */

/* internal: one dot of length HD */
__device__ __forceinline__ float dot64(const float* a, const float* b) {
  float s = 0.0f;
#pragma unroll
  for (int l = 0; l < HD; l++) s = fmaf(a[l], b[l], s);
  return s;
}

template <int IT>
__global__ void attn_kernel(const float* __restrict__ q, const float* __restrict__ k,
                            const float* __restrict__ v, int t, int d, int hd, int heads,
                            float* __restrict__ out, int CH) {
  int h = blockIdx.y;
  int i0 = blockIdx.x * IT;
  int ni = min(IT, t - i0);
  if (ni <= 0) return;
  int tid = threadIdx.x;

  extern __shared__ float sh[];
  float* qs = sh;                       /* IT*HDP : q rows of this head */
  float* ss = qs + IT * HDP;             /* IT*t   : scores / probabilities */
  float* kc = ss + IT * t;               /* CH*HDP : k (or v) chunk */
  float* red = kc + HDP * CH;            /* NTH    : block reductions */

  /* ---- load q tile (head h) ---- */
  for (int idx = tid; idx < ni * HD; idx += NTH) {
    int i = idx >> 6, l = idx & 63;
    qs[i * HDP + l] = q[(size_t)(i0 + i) * d + h * hd + l];
  }
  __syncthreads();

  /* ---- scores ---- */
  for (int j0 = 0; j0 < t; j0 += CH) {
    int nj = min(CH, t - j0);
    for (int idx = tid; idx < nj * HD; idx += NTH) {
      int j = idx >> 6, l = idx & 63;
      kc[j * HDP + l] = k[(size_t)(j0 + j) * d + h * hd + l];
    }
    __syncthreads();
    for (int idx = tid; idx < ni * nj; idx += NTH) {
      int i = idx / nj, j = idx - i * nj;
      ss[i * t + j0 + j] = dot64(qs + i * HDP, kc + j * HDP) * 0.125f;
    }
    __syncthreads();
  }

  /* ---- softmax per row, warp-level ---- */
  {
    int warp = tid >> 5, lane = tid & 31, nwarp = NTH >> 5;
    for (int i = warp; i < ni; i += nwarp) {
      float* srow = ss + (size_t)i * t;
      float m = -1e30f;
      for (int j = lane; j < t; j += 32) m = fmaxf(m, srow[j]);
      for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
      float s = 0.0f;
      for (int j = lane; j < t; j += 32) {
        float e = __expf(srow[j] - m);
        srow[j] = e;
        s += e;
      }
      for (int off = 16; off; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
      float inv = 1.0f / s;
      for (int j = lane; j < t; j += 32) srow[j] *= inv;
    }
  }
  __syncthreads();

  /* ---- PV: register accumulators, one per (i,lane) pair ---- */
  constexpr int npairs = (IT * HD + NTH - 1) / NTH;   /* compile-time: keeps acc[] in registers */
  float acc[npairs];
#pragma unroll
  for (int p = 0; p < npairs; p++) acc[p] = 0.0f;

  for (int j0 = 0; j0 < t; j0 += CH) {
    int nj = min(CH, t - j0);
    for (int idx = tid; idx < nj * HD; idx += NTH) {
      int j = idx >> 6, l = idx & 63;
      kc[j * HDP + l] = v[(size_t)(j0 + j) * d + h * hd + l];
    }
    __syncthreads();
    for (int p = 0; p < npairs; p++) {
      int idx = tid + p * NTH;
      if (idx >= ni * HD) break;
      int i = idx >> 6, lane = idx & 63;
      const float* prow = ss + (size_t)i * t + j0;
      float a = acc[p];
      for (int j = 0; j < nj; j++) a = fmaf(prow[j], kc[j * HDP + lane], a);
      acc[p] = a;
    }
    __syncthreads();
  }

  for (int p = 0; p < npairs; p++) {
    int idx = tid + p * NTH;
    if (idx >= ni * HD) break;
    int i = idx >> 6, lane = idx & 63;
    out[(size_t)(i0 + i) * d + h * hd + lane] = acc[p];
  }
}

/* ------------------------------------------------------------------ wrapper */
static int g_smem_attr_set = 0;

template <int IT>
static void launch(const float* q, const float* k, const float* v, int t, int d, int hd,
                   int heads, float* out, cudaStream_t s, int CH) {
  size_t nb = ((size_t)IT * HDP + (size_t)IT * t + (size_t)CH * HDP + NTH) * sizeof(float);
  if (!g_smem_attr_set) {
    cudaFuncSetAttribute(attn_kernel<IT>, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
    g_smem_attr_set = 1;
  }
  dim3 grid((t + IT - 1) / IT, heads);
  attn_kernel<IT><<<grid, NTH, nb, s>>>(q, k, v, t, d, hd, heads, out, CH);
}

extern "C" int wk_attention_tiled(void* stream, const float* q, const float* k, const float* v,
                                 int t, int d, int heads, float* scratch, float* out) {
  (void)scratch;
  if (!q || !k || !v || !out) return -1;
  if (t <= 0 || d <= 0 || heads <= 0) return 0;
  if (d % heads != 0) return -2;
  int hd = d / heads;
  if (hd != HD) return -3; /* this build is specialised for head_dim = 64 */
  cudaStream_t s = (cudaStream_t)stream;
  int CH = 64; /* j chunk */
  if (t <= 192) launch<32>(q, k, v, t, d, hd, heads, out, s, CH);
  else launch<16>(q, k, v, t, d, hd, heads, out, s, CH);
  return 0;
}

/* same, but with an explicit i-tile size (for benchmarking) */
extern "C" int wk_attention_tiled_tuned(void* stream, const float* q, const float* k,
                                       const float* v, int t, int d, int heads, float* out,
                                       int it, int ch) {
  if (!q || !k || !v || !out) return -1;
  int hd = d / heads;
  if (hd != HD) return -3;
  cudaStream_t s = (cudaStream_t)stream;
  if (ch <= 0) ch = 64;
  switch (it) {
    case 8: launch<8>(q, k, v, t, d, hd, heads, out, s, ch); break;
    case 16: launch<16>(q, k, v, t, d, hd, heads, out, s, ch); break;
    case 32: launch<32>(q, k, v, t, d, hd, heads, out, s, ch); break;
    case 64: launch<64>(q, k, v, t, d, hd, heads, out, s, ch); break;
    default: return -4;
  }
  return 0;
}

extern "C" const char* wk_attention_tiled_version(void) {
  return "wk_attention_tiled 1.0 (sm_120, fp32, IT in {8,16,32,64}, hd=64)";
}
