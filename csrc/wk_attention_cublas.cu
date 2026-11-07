/* wk_attention_cublas.cu -- multi-head attention as "2 x cublasSgemmStridedBatched
 * + one softmax kernel", i.e. the classic formulation.
 *
 *   out[i][h*hd + lane] = sum_j softmax_j( q_i[h] . k_j[h] * 0.125 ) * v_j[h][lane]
 *
 * q/k/v/out are [t][d] row-major f32, heads = 8, hd = 64.
 *
 * LAYOUT DERIVATION (this is where a hand-rolled cuBLAS version usually goes wrong)
 * ------------------------------------------------------------------------------
 * GEMM1 computes the score matrix TRANSPOSED,  P^T[j][i] = 0.125 * k_j . q_i :
 *   op(T, N), m=t, n=t, k=64
 *   A stored 64 x t col-major, lda=d=512, strideA=64 -> A(x,j) = k[j*512 + h*64 + x]
 *   B stored 64 x t col-major, ldb=d=512, strideB=64 -> B(x,i) = q[i*512 + h*64 + x]
 *   C stored t x t col-major, ldc=t, strideC=t*t     -> C(j,i) = scratch[h*t*t + j + i*t]
 *   => P[i][j] lives at scratch[h*t*t + j + i*t], i.e. row i of P is CONTIGUOUS.
 *   Keeping P^T (instead of P) is what makes the softmax fast: it reduces along
 *   contiguous memory and each thread's row stays in L1 across both passes.
 *
 * softmax: 2-pass online softmax over the contiguous t-element row,
 *          grid=(ceil(t/32), heads), block=(8,32).
 *
 * GEMM2: out^T = V^T . P^T, i.e. C(lane,i) = sum_j v[j][lane] * P[i][j] :
 *   op(N, N), m=hd=64, n=t, k=t
 *   A stored 64 x t col-major, lda=d=512, strideA=64 -> A(lane,j) = v[j*512 + h*64 + lane]
 *   B stored t x t col-major, ldb=t, strideB=t*t     -> B(j,i) = scratch[h*t*t + j + i*t]
 *   C stored 64 x t col-major, ldc=d=512, strideC=64 -> C(lane,i) = out[i*512 + h*64 + lane]
 *   C is EXACTLY row-major out[t][512].  The naive version
 *       sgemmStridedBatched(N, N, m=t, n=64, k=t, A=scratch, lda=t, B=v, ldb=512,
 *                           C=out, ldc=512, batch=heads)
 *   is wrong twice: (a) its B(l,j) = v + h*64 + l + j*512 reads v transposed,
 *   (b) its C(i,lane) = out + i + lane*512 writes the transpose of the result.
 *   Both come from cuBLAS being column-major: with op(N,N) the output is a t x 64
 *   column-major matrix whose element (i,lane) sits at i + lane*512, not at
 *   i*512 + lane.  Swapping to the (V^T, P^T) order fixes both at once.
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <mutex>
#include <cstdlib>

#define ATTN_HD 64
#define SOF_LANES 8

/* 2-pass online softmax over the contiguous last dim; row (head h, i) starts at
 * scratch + h*t*t + i*t.  block = (SOF_LANES, 32) threads -> 32 rows per block. */
__global__ void attn_softmax_rows(const float* __restrict__ /*unused*/, float* __restrict__ s, int t) {
  int jl = threadIdx.x;
  int i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i >= t) return;
  float* r = s + ((size_t)blockIdx.y * t + i) * t;

  /* pass 1: online max + sum (one read of the row) */
  float m = -1e30f, sum = 0.0f;
  for (int j = jl; j < t; j += SOF_LANES) {
    float v = r[j];
    float nm = fmaxf(m, v);
    sum = sum * __expf(m - nm) + __expf(v - nm);
    m = nm;
  }
  /* combine the 8 lanes of this row (same warp -> shuffle only, width=8) */
  for (int off = SOF_LANES / 2; off; off >>= 1) {
    float m2 = __shfl_xor_sync(0xffffffffu, m, off, SOF_LANES);
    float s2 = __shfl_xor_sync(0xffffffffu, sum, off, SOF_LANES);
    float nm = fmaxf(m, m2);
    sum = sum * __expf(m - nm) + s2 * __expf(m2 - nm);
    m = nm;
  }
  float inv = 1.0f / sum;

  /* pass 2: write the normalised probabilities (one read + one write) */
  for (int j = jl; j < t; j += SOF_LANES) r[j] = __expf(r[j] - m) * inv;
}

static cublasHandle_t g_h = nullptr;
static std::mutex g_mu;

extern "C" int wk_attention_cublas(void* stream, const float* q, const float* k, const float* v,
                                   int t, int d, int heads, float* scratch, float* out) {
  if (!q || !k || !v || !out || !scratch) return -1;
  if (t <= 0 || heads <= 0 || d % heads != 0) return -2;
  int hd = d / heads;
  if (hd != ATTN_HD) return -3;
  cudaStream_t s = (cudaStream_t)stream;

  std::lock_guard<std::mutex> lk(g_mu);
  if (!g_h) {
    if (cublasCreate(&g_h) != CUBLAS_STATUS_SUCCESS) return -4;
    const char* m = getenv("WTL_ATTN_MATH");   /* "tf32" -> tensor-core TF32 (faster, ~1e-3) */
    cublasSetMathMode(g_h, (m && m[0] == 't' && m[1] == 'f') ? CUBLAS_TF32_TENSOR_OP_MATH
                                                            : CUBLAS_DEFAULT_MATH);
  }
  cublasSetStream(g_h, s);

  const float alpha = 0.125f, beta = 0.0f;
  /* GEMM1: P^T[h] = 0.125 * K[h] * Q[h]^T   (t x t col-major, ld=t) */
  cublasStatus_t st = cublasSgemmStridedBatched(
      g_h, CUBLAS_OP_T, CUBLAS_OP_N, t, t, ATTN_HD,
      &alpha, k, d, 64L, q, d, 64L,
      &beta, scratch, t, (long long)t * t, heads);
  if (st != CUBLAS_STATUS_SUCCESS) return -5;

  {
    dim3 block(SOF_LANES, 32);
    dim3 grid((t + 31) / 32, heads);
    attn_softmax_rows<<<grid, block, 0, s>>>(nullptr, scratch, t);
  }

  /* GEMM2: out^T[h] = V^T[h] * P^T[h]   (64 x t col-major, ld=512) */
  const float one = 1.0f;
  st = cublasSgemmStridedBatched(
      g_h, CUBLAS_OP_N, CUBLAS_OP_N, ATTN_HD, t, t,
      &one, v, d, 64L, scratch, t, (long long)t * t,
      &beta, out, d, 64L, heads);
  if (st != CUBLAS_STATUS_SUCCESS) return -6;
  return 0;
}

/* ---- parts, for benchmarking only (not needed by the engine) -------------- */
extern "C" int wk_attention_cublas_part(int part, void* stream, const float* q, const float* k,
                                        const float* v, int t, int d, int heads, float* scratch,
                                        float* out) {
  cudaStream_t s = (cudaStream_t)stream;
  if (!g_h) { if (cublasCreate(&g_h) != CUBLAS_STATUS_SUCCESS) return -4; }
  cublasSetStream(g_h, s);
  const float alpha = 0.125f, beta = 0.0f, one = 1.0f;
  if (part == 0)
    return cublasSgemmStridedBatched(g_h, CUBLAS_OP_T, CUBLAS_OP_N, t, t, ATTN_HD, &alpha, k, d,
                                     64L, q, d, 64L, &beta, scratch, t, (long long)t * t, heads);
  if (part == 1) {
    dim3 block(SOF_LANES, 32);
    dim3 grid((t + 31) / 32, heads);
    attn_softmax_rows<<<grid, block, 0, s>>>(nullptr, scratch, t);
    return 0;
  }
  return cublasSgemmStridedBatched(g_h, CUBLAS_OP_N, CUBLAS_OP_N, ATTN_HD, t, t, &one, v, d, 64L,
                                   scratch, t, (long long)t * t, &beta, out, d, 64L, heads);
}
