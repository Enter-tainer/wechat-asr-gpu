/* wtlgpu.cu -- int8 GEMM primitive + CUDA-graph helpers for the wtl ASR engine
 *              on sm_120 (RTX 5060 Ti).
 *
 *   out[t][n] f32 = ( int32 dot(xq[t][k], w[n][k]) ) * xs[t] * ws + bias[n]
 *
 * cuBLASLt INT8 tensor-core GEMM (CUBLAS_COMPUTE_32I) + fused epilogue kernels.
 *
 * cuBLASLt framing:
 *   C[n,t] = op(A) * op(B)
 *     A = w  : row-major [n][k] -> col-major KxN (lda=k), opA = T   (cublas M = n)
 *     B = xq : row-major [t][k] -> col-major KxT (ldb=k), opB = N   (cublas N = t)
 *     C      : col-major [n][t] (ldc=n), CUDA_R_32I
 *   The int32 accumulator's linear layout is therefore identical to row-major [t][n],
 *   so the epilogue is a pure element-wise pass.
 *
 * cuBLASLt int8 requires the GEMM M dimension (= n here) to be a multiple of 4.
 * The vocabulary layer has n = 10502 (not a multiple of 4); for such n the library
 * transparently zero-pads the weight rows to align4(n) once (cached), runs the GEMM
 * at n', and the epilogue writes only the first n columns (row stride n').
 *
 * Build:
 *   /usr/local/cuda-13.0/bin/nvcc -O3 -arch=sm_120 -shared -Xcompiler -fPIC \
 *     -o libwtlgpu.so wtlgpu.cu -L/usr/local/cuda-13.0/lib64 -I/usr/local/cuda-13.0/include \
 *     -lcublasLt -lcublas -lcudart -Xlinker -rpath -Xlinker /usr/local/cuda-13.0/lib64
 */
#include "wtlgpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mutex>

#include <cuda_runtime.h>
#include <cublasLt.h>

#define WTL_MAX_CACHE 512
#define WTL_MAX_PAD 4
#define WTL_SCRATCH_BYTES (24ull * 1024 * 1024) /* initial int32 acc scratch, grows on demand */

struct CacheEntry {
  int k = 0, n = 0, t = 0;
  bool valid = false;
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr;
  cublasLtMatmulAlgo_t algo{};
  int n_algo = 0;
};

/* zero-padded weight copy for n % 4 != 0 (e.g. the 10502-wide vocabulary layer) */
struct PadEntry {
  bool valid = false;
  size_t w_off = 0;
  int n = 0, k = 0;
  void* buf = nullptr;
  size_t bytes = 0;
};

struct WpCtx {
  cublasLtHandle_t lt = nullptr;
  cudaStream_t call_stream = nullptr;   /* NULL == legacy default stream */
  cudaStream_t graph_stream = nullptr;  /* private non-blocking stream, used once capture starts */
  cudaStream_t search_stream = nullptr; /* private, algorithm search only */
  int capturing = 0;
  int last_err = 0;          /* reason for the last get_entry() failure (-3 no algo / -19 cache full) */
  size_t last_graph_nodes = 0;
  void* w = nullptr;
  size_t w_bytes = 0;
  void* acc = nullptr;
  size_t acc_bytes = 0;
  void* ws = nullptr;
  size_t ws_bytes = 0;
  std::mutex mu;
  CacheEntry cache[WTL_MAX_CACHE];
  int n_cache = 0;
  PadEntry pad[WTL_MAX_PAD];
  int n_pad = 0;
};

/* ------------------------------------------------------------------ helpers */
__device__ __forceinline__ float wtl_gelu(float x) {
  return 0.5f * x * (1.0f + tanhf(0.7978846f * (x + 0.044715f * x * x * x)));
}

/* ------------------------------------------------------------------ kernels */
/* f32 epilogue: out[t][n] = acc[t][stride] * xs[row] * ws + bias[col] */
__global__ void wp_epilogue(const int32_t* __restrict__ acc, int stride, const float* __restrict__ xs,
                            int t, int n, float ws, const float* __restrict__ bias,
                            float* __restrict__ out) {
  int row = blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= t || col >= n) return;
  out[(size_t)row * n + col] =
      (float)acc[(size_t)row * stride + col] * (xs[row] * ws) + (bias ? bias[col] : 0.0f);
}

/* fused epilogue + optional GELU + per-row dynamic quantisation (Rust `quant()`):
 *   y = acc*xs[row]*ws + bias;  g = GELU ? gelu(y) : y
 *   s = max|g|/127 (or 1.0);    q = round(g/s) clamped
 * fast path: n % blockDim == 0 && n/blockDim <= 16 -> values kept in registers */
template <bool GELU>
__global__ void wp_epilogue_q_fast(const int32_t* __restrict__ acc, int stride,
                                   const float* __restrict__ xs, int t, int n, float ws,
                                   const float* __restrict__ bias, int8_t* __restrict__ q,
                                   float* __restrict__ s) {
  int row = blockIdx.x;
  if (row >= t) return;
  __shared__ float red[256];
  int tid = threadIdx.x, nth = blockDim.x;
  float scale = xs[row] * ws;
  const int32_t* ar = acc + (size_t)row * stride;
  float y[16];
  int cnt = n / nth;
  if (cnt > 16) cnt = 16;
  float m = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    if (i >= cnt) break;
    int c = tid + i * nth;
    float v = (float)ar[c] * scale + (bias ? bias[c] : 0.0f);
    if (GELU) v = wtl_gelu(v);
    y[i] = v;
    m = fmaxf(m, fabsf(v));
  }
  red[tid] = m;
  __syncthreads();
  for (int st = nth >> 1; st > 0; st >>= 1) {
    if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
    __syncthreads();
  }
  float mx = red[0];
  float sc = (mx > 0.0f) ? mx / 127.0f : 1.0f;
  float inv = 1.0f / sc;
  int8_t* qr = q + (size_t)row * n;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    if (i >= cnt) break;
    float v = roundf(y[i] * inv);
    v = fminf(127.0f, fmaxf(-127.0f, v));
    qr[tid + i * nth] = (int8_t)v;
  }
  if (tid == 0) s[row] = sc;
}

/* generic fallback: values staged in dynamic shared memory (any n) */
template <bool GELU>
__global__ void wp_epilogue_q_gen(const int32_t* __restrict__ acc, int stride,
                                  const float* __restrict__ xs, int t, int n, float ws,
                                  const float* __restrict__ bias, int8_t* __restrict__ q,
                                  float* __restrict__ s) {
  int row = blockIdx.x;
  if (row >= t) return;
  extern __shared__ float sh[];
  __shared__ float red[256];
  int tid = threadIdx.x, nth = blockDim.x;
  float scale = xs[row] * ws;
  float m = 0.0f;
  for (int c = tid; c < n; c += nth) {
    float v = (float)acc[(size_t)row * stride + c] * scale + (bias ? bias[c] : 0.0f);
    if (GELU) v = wtl_gelu(v);
    sh[c] = v;
    m = fmaxf(m, fabsf(v));
  }
  red[tid] = m;
  __syncthreads();
  for (int st = nth >> 1; st > 0; st >>= 1) {
    if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
    __syncthreads();
  }
  float mx = red[0];
  float sc = (mx > 0.0f) ? mx / 127.0f : 1.0f;
  float inv = 1.0f / sc;
  for (int c = tid; c < n; c += nth) {
    float v = roundf(sh[c] * inv);
    v = fminf(127.0f, fmaxf(-127.0f, v));
    q[(size_t)row * n + c] = (int8_t)v;
  }
  if (tid == 0) s[row] = sc;
}

/* standalone per-row dynamic quantisation of an f32 [t][n] buffer (Rust `quant()`),
 * same rounding as the fused epilogue.  Needs n % 256 == 0 && n/256 <= 16. */
__global__ void wp_quant_rows_k(const float* __restrict__ y, int t, int n,
                                int8_t* __restrict__ q, float* __restrict__ s) {
  int row = blockIdx.x;
  if (row >= t) return;
  __shared__ float red[256];
  int tid = threadIdx.x, nth = blockDim.x;
  const float* yr = y + (size_t)row * n;
  float v[16];
  int cnt = n / nth;
  if (cnt > 16) cnt = 16;
  float m = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    if (i >= cnt) break;
    v[i] = yr[tid + i * nth];
    m = fmaxf(m, fabsf(v[i]));
  }
  red[tid] = m;
  __syncthreads();
  for (int st = nth >> 1; st > 0; st >>= 1) {
    if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
    __syncthreads();
  }
  float mx = red[0];
  float sc = (mx > 0.0f) ? mx / 127.0f : 1.0f;
  float inv = 1.0f / sc;
  int8_t* qr = q + (size_t)row * n;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    if (i >= cnt) break;
    float qv = roundf(v[i] * inv);
    qv = fminf(127.0f, fmaxf(-127.0f, qv));
    qr[tid + i * nth] = (int8_t)qv;
  }
  if (tid == 0) s[row] = sc;
}

static const char* cstat(cublasStatus_t s) {
  switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "OK";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "EXEC_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL";
    case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    default: return "OTHER";
  }
}

/* ------------------------------------------------------- algorithm selection */
static bool build_entry(WpCtx* c, CacheEntry* e, int k, int n, int t) {
  e->k = k; e->n = n; e->t = t;
  cublasStatus_t st = cublasLtMatmulDescCreate(&e->desc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
  if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "[wtlgpu] descCreate %s\n", cstat(st)); return false; }
  {
    cublasOperation_t opA = CUBLAS_OP_T, opB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(e->desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(e->desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
  }
  if (cublasLtMatrixLayoutCreate(&e->Ad, CUDA_R_8I, k, n, k) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&e->Bd, CUDA_R_8I, k, t, k) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&e->Cd, CUDA_R_32I, n, t, n) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "[wtlgpu] layoutCreate failed\n");
    return false;
  }
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &c->ws_bytes, sizeof(c->ws_bytes));
  cublasLtMatmulHeuristicResult_t heur[16];
  int nres = 0;
  st = cublasLtMatmulAlgoGetHeuristic(c->lt, e->desc, e->Ad, e->Bd, e->Cd, e->Cd, pref, 16, heur, &nres);
  cublasLtMatmulPreferenceDestroy(pref);
  if (st != CUBLAS_STATUS_SUCCESS || nres == 0) {
    fprintf(stderr, "[wtlgpu] no int8 algo for k=%d n=%d t=%d (%s n=%d)\n", k, n, t, cstat(st), nres);
    return false;
  }
  e->n_algo = nres;

  cudaStream_t ss = c->search_stream;
  int8_t* dummy_x = nullptr;
  if (cudaMalloc(&dummy_x, (size_t)k * t) != cudaSuccess) return false;
  cudaMemsetAsync(dummy_x, 1, (size_t)k * t, ss);
  struct Cand { cublasLtMatmulAlgo_t a; double us; };
  std::vector<Cand> cands;
  int32_t alpha = 1, beta = 0;
  for (int i = 0; i < nres && (int)cands.size() < 6; i++) {
    if (heur[i].state != CUBLAS_STATUS_SUCCESS) continue;
    if (heur[i].workspaceSize > c->ws_bytes) continue;
    cublasLtMatmulAlgo_t a = heur[i].algo;
    auto fn = [&]() {
      cublasLtMatmul(c->lt, e->desc, &alpha, c->w, e->Ad, dummy_x, e->Bd, &beta,
                     c->acc, e->Cd, c->acc, e->Cd, &a, c->ws, c->ws_bytes, ss);
    };
    for (int r = 0; r < 2; r++) fn();
    cudaStreamSynchronize(ss);
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    if (cudaStreamBeginCapture(ss, cudaStreamCaptureModeThreadLocal) != cudaSuccess) break;
    for (int r = 0; r < 4; r++) fn();
    if (cudaStreamEndCapture(ss, &graph) != cudaSuccess) break;
    if (cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) break;
    double best = 1e30;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < 3; r++) {
      cudaEventRecord(e0, ss); cudaGraphLaunch(exec, ss); cudaEventRecord(e1, ss);
      cudaStreamSynchronize(ss);
      float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
      best = std::min<double>(best, ms * 1000.0 / 4.0);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    cands.push_back({a, best});
  }
  cudaFree(dummy_x);
  if (cands.empty()) { fprintf(stderr, "[wtlgpu] no usable algo k=%d n=%d t=%d\n", k, n, t); return false; }
  std::sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) { return x.us < y.us; });
  e->algo = cands[0].a;
  e->valid = true;
  return true;
}

static bool stream_capturing(cudaStream_t s) {
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  if (s) cudaStreamIsCapturing(s, &cs);
  return cs != cudaStreamCaptureStatusNone;
}

static CacheEntry* get_entry(WpCtx* c, int k, int n, int t, cudaStream_t stream) {
  for (int i = 0; i < c->n_cache; i++) {
    CacheEntry* e = &c->cache[i];
    if (e->valid && e->k == k && e->n == n && e->t == t) return e;
  }
  if (stream_capturing(stream)) {
    fprintf(stderr, "[wtlgpu] shape k=%d n=%d t=%d not warmed up before graph capture\n", k, n, t);
    return nullptr;
  }
  if (c->n_cache >= WTL_MAX_CACHE) {
    fprintf(stderr, "[wtlgpu] shape cache full (%d entries) for k=%d n=%d t=%d\n",
            WTL_MAX_CACHE, k, n, t);
    c->last_err = -19;
    return nullptr;
  }
  CacheEntry* e = &c->cache[c->n_cache];
  if (!build_entry(c, e, k, n, t)) return nullptr;
  c->n_cache++;
  return e;
}

/* padded weight copy for n % 4 != 0 (cuBLASLt int8 needs GEMM M % 4 == 0) */
static const int8_t* find_pad(WpCtx* c, size_t w_off, int n, int k) {
  for (int i = 0; i < c->n_pad; i++) {
    PadEntry* p = &c->pad[i];
    if (p->valid && p->w_off == w_off && p->n == n && p->k == k) return (const int8_t*)p->buf;
  }
  return nullptr;
}

static const int8_t* padded_weight(WpCtx* c, size_t w_off, int n, int k, int n_pad, int* err) {
  if (c->n_pad >= WTL_MAX_PAD) { *err = -8; return nullptr; }
  PadEntry* p = &c->pad[c->n_pad];
  size_t bytes = (size_t)n_pad * k;
  if (cudaMalloc(&p->buf, bytes) != cudaSuccess) { *err = -9; return nullptr; }
  if (cudaMemcpy(p->buf, (const char*)c->w + w_off, (size_t)n * k, cudaMemcpyDeviceToDevice) != cudaSuccess) {
    cudaFree(p->buf); p->buf = nullptr; *err = -9; return nullptr;
  }
  if (cudaMemset((char*)p->buf + (size_t)n * k, 0, (size_t)(n_pad - n) * k) != cudaSuccess) {
    cudaFree(p->buf); p->buf = nullptr; *err = -9; return nullptr;
  }
  p->valid = true; p->w_off = w_off; p->n = n; p->k = k; p->bytes = bytes;
  c->n_pad++;
  return (const int8_t*)p->buf;
}

/* --------------------------------------------------------- one GEMM + epilogue
 * mode 0: f32 out;  mode 1: per-row quant -> int8;  mode 2: GELU + per-row quant */
static int gemm_impl(WpCtx* c, const int8_t* xq, const float* xs, int t, int k,
                     size_t w_off, int n, float ws, const float* bias_dev,
                     cudaStream_t stream, float* out, int8_t* out_q, float* out_s, int mode) {
  if (!c || !xq || !xs || (out_q ? !out_s : !out)) return -1;
  if (t <= 0 || k <= 0 || n <= 0) return 0;
  if ((size_t)n * (size_t)k > c->w_bytes || w_off > c->w_bytes - (size_t)n * (size_t)k) return -2;

  std::lock_guard<std::mutex> lock(c->mu);

  int np = (n + 3) & ~3;      /* cuBLASLt int8 requires M % 4 == 0 */
  int err = 0;
  const int8_t* A;
  if (np == n) {
    A = (const int8_t*)c->w + w_off;
  } else {
    A = find_pad(c, w_off, n, k);
    if (!A) {
      if (stream_capturing(stream)) {
        fprintf(stderr, "[wtlgpu] n=%d needs padding; warm it up before the graph capture\n", n);
        return -6;
      }
      A = padded_weight(c, w_off, n, k, np, &err);
      if (!A) return err;
    }
  }

  size_t acc_need = (size_t)t * np * 4;
  if (acc_need > c->acc_bytes) {
    if (stream_capturing(stream)) return -5;
    size_t want = acc_need + (acc_need >> 1);
    void* na = nullptr;
    if (cudaMalloc(&na, want) != cudaSuccess) return -5;
    if (c->acc) cudaFree(c->acc);
    c->acc = na; c->acc_bytes = want;
  }

  c->last_err = -3;
  CacheEntry* e = get_entry(c, k, np, t, stream);
  if (!e) return stream_capturing(stream) ? -6 : c->last_err;

  const int32_t alpha = 1, beta = 0;
  cublasStatus_t st = cublasLtMatmul(c->lt, e->desc, &alpha, A, e->Ad, xq, e->Bd, &beta,
                                     c->acc, e->Cd, c->acc, e->Cd, &e->algo,
                                     c->ws, c->ws_bytes, stream);
  if (st != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "[wtlgpu] cublasLtMatmul failed: %s (k=%d n=%d t=%d)\n", cstat(st), k, n, t);
    return -4;
  }
  if (mode == 0) {
    int threads = 256;
    dim3 grid((unsigned)((n + threads - 1) / threads), (unsigned)t);
    wp_epilogue<<<grid, threads, 0, stream>>>((const int32_t*)c->acc, np, xs, t, n, ws, bias_dev, out);
  } else {
    int threads = 256;
    bool fast = (n % threads == 0 && n / threads <= 16);
    if (mode == 1) {
      if (fast) wp_epilogue_q_fast<false><<<(unsigned)t, threads, 0, stream>>>(
          (const int32_t*)c->acc, np, xs, t, n, ws, bias_dev, out_q, out_s);
      else {
        size_t smem = (size_t)n * sizeof(float);
        cudaFuncSetAttribute(wp_epilogue_q_gen<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        wp_epilogue_q_gen<false><<<(unsigned)t, threads, smem, stream>>>(
            (const int32_t*)c->acc, np, xs, t, n, ws, bias_dev, out_q, out_s);
      }
    } else {
      if (fast) wp_epilogue_q_fast<true><<<(unsigned)t, threads, 0, stream>>>(
          (const int32_t*)c->acc, np, xs, t, n, ws, bias_dev, out_q, out_s);
      else {
        size_t smem = (size_t)n * sizeof(float);
        cudaFuncSetAttribute(wp_epilogue_q_gen<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        wp_epilogue_q_gen<true><<<(unsigned)t, threads, smem, stream>>>(
            (const int32_t*)c->acc, np, xs, t, n, ws, bias_dev, out_q, out_s);
      }
    }
  }
  return 0;
}

/* --------------------------------------------------------------------- API */
extern "C" {

const char* wp_version(void) {
  return "wtlgpu 1.3 (cublasLt int8 sm_120; wp_gemm/2/q/qg; auto n-pad; cuda-graph)";
}

WpCtx* wp_create(const void* w_host, size_t bytes) {
  WpCtx* c = new WpCtx();
  if (cublasLtCreate(&c->lt) != CUBLAS_STATUS_SUCCESS) { delete c; return nullptr; }
  if (cudaStreamCreateWithFlags(&c->search_stream, cudaStreamNonBlocking) != cudaSuccess) {
    cublasLtDestroy(c->lt); delete c; return nullptr;
  }
  if (cudaStreamCreateWithFlags(&c->graph_stream, cudaStreamNonBlocking) != cudaSuccess) {
    cudaStreamDestroy(c->search_stream); cublasLtDestroy(c->lt); delete c; return nullptr;
  }
  c->call_stream = nullptr; /* legacy default stream: implicitly ordered with stream-0 kernels */
  c->ws_bytes = 64ull * 1024 * 1024;
  if (cudaMalloc(&c->ws, c->ws_bytes) != cudaSuccess) { wp_destroy(c); return nullptr; }
  c->acc_bytes = WTL_SCRATCH_BYTES;
  if (cudaMalloc(&c->acc, c->acc_bytes) != cudaSuccess) { wp_destroy(c); return nullptr; }
  c->w_bytes = bytes;
  if (bytes > 0) {
    if (cudaMalloc(&c->w, bytes) != cudaSuccess) { wp_destroy(c); return nullptr; }
    if (w_host) {
      if (cudaMemcpy(c->w, w_host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { wp_destroy(c); return nullptr; }
    } else {
      cudaMemset(c->w, 0, bytes);
    }
  }
  return c;
}

void wp_destroy(WpCtx* c) {
  if (!c) return;
  cudaDeviceSynchronize();
  for (int i = 0; i < c->n_cache; i++) {
    CacheEntry* e = &c->cache[i];
    if (e->desc) cublasLtMatmulDescDestroy(e->desc);
    if (e->Ad) cublasLtMatrixLayoutDestroy(e->Ad);
    if (e->Bd) cublasLtMatrixLayoutDestroy(e->Bd);
    if (e->Cd) cublasLtMatrixLayoutDestroy(e->Cd);
  }
  for (int i = 0; i < c->n_pad; i++) if (c->pad[i].buf) cudaFree(c->pad[i].buf);
  if (c->w) cudaFree(c->w);
  if (c->acc) cudaFree(c->acc);
  if (c->ws) cudaFree(c->ws);
  if (c->graph_stream) cudaStreamDestroy(c->graph_stream);
  if (c->search_stream) cudaStreamDestroy(c->search_stream);
  if (c->lt) cublasLtDestroy(c->lt);
  delete c;
}

void* wp_alloc(size_t bytes) {
  void* p = nullptr;
  if (bytes == 0) return nullptr;
  return cudaMalloc(&p, bytes) == cudaSuccess ? p : nullptr;
}
void wp_free(void* p) { if (p) cudaFree(p); }

int wp_h2d(void* dev, const void* host, size_t bytes) {
  if (!dev || !host) return -1;
  return cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -2;
}
int wp_d2h(void* host, const void* dev, size_t bytes) {
  if (!host || !dev) return -1;
  return cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -2;
}
void* wp_devptr(WpCtx* c, size_t off) {
  if (!c || off > c->w_bytes) return nullptr;
  return (char*)c->w + off;
}
int wp_sync(WpCtx* c) {
  if (!c) return -1;
  cudaStream_t s = c->call_stream ? c->call_stream : nullptr;
  return cudaStreamSynchronize(s) == cudaSuccess ? 0 : -2;
}
int wp_set_stream(WpCtx* c, void* stream) {
  if (!c) return -1;
  if (c->capturing) return -16;
  c->call_stream = (cudaStream_t)stream;
  return 0;
}
void* wp_stream(WpCtx* c) { return c ? (void*)c->call_stream : nullptr; }
size_t wp_scratch_bytes(WpCtx* c) { return c ? c->acc_bytes : 0; }

int wp_gemm(WpCtx* c, const int8_t* xq, const float* xs, int t, int k,
            size_t w_off, int n, float ws, size_t bias_off, float* out) {
  const float* bias = nullptr;
  if (c && bias_off != (size_t)-1) {
    if (bias_off + (size_t)n * 4 > c->w_bytes) return -2;
    bias = (const float*)((const char*)c->w + bias_off);
  }
  return gemm_impl(c, xq, xs, t, k, w_off, n, ws, bias, c ? c->call_stream : nullptr, out, nullptr, nullptr, 0);
}

int wp_gemm2(WpCtx* c, const int8_t* xq, const float* xs, int t, int k,
             size_t w_off, int n, float ws, const float* bias_dev, void* stream, float* out) {
  return gemm_impl(c, xq, xs, t, k, w_off, n, ws, bias_dev, (cudaStream_t)stream, out, nullptr, nullptr, 0);
}

int wp_gemm_q(WpCtx* c, const int8_t* xq, const float* xs, int t, int k,
              size_t w_off, int n, float ws, const float* bias_dev, void* stream,
              int8_t* out_q, float* out_s) {
  return gemm_impl(c, xq, xs, t, k, w_off, n, ws, bias_dev, (cudaStream_t)stream, nullptr, out_q, out_s, 1);
}

int wp_gemm_qg(WpCtx* c, const int8_t* xq, const float* xs, int t, int k,
               size_t w_off, int n, float ws, const float* bias_dev, void* stream,
               int8_t* out_q, float* out_s) {
  return gemm_impl(c, xq, xs, t, k, w_off, n, ws, bias_dev, (cudaStream_t)stream, nullptr, out_q, out_s, 2);
}

int wp_quant_rows(const float* y, int t, int n, void* stream, int8_t* out_q, float* out_s) {
  if (!y || !out_q || !out_s) return -1;
  if (t <= 0 || n <= 0) return 0;
  if (n % 256 != 0 || n / 256 > 16) return -7;
  wp_quant_rows_k<<<(unsigned)t, 256, 0, (cudaStream_t)stream>>>(y, t, n, out_q, out_s);
  return 0;
}

/* ------------------------------------------------------------ CUDA graphs */
int wp_capture_begin(WpCtx* c) {
  if (!c) return -1;
  if (c->capturing) { fprintf(stderr, "[wtlgpu] capture already in progress\n"); return -10; }
  if (!c->graph_stream) return -10;
  /* The legacy default stream cannot be captured: finish whatever is queued on it,
   * then switch the context to the private non-blocking stream for good (wp_stream
   * reports it from now on, so caller kernels land in the same graph). */
  if (c->call_stream == nullptr || c->call_stream == (cudaStream_t)0) {
    cudaError_t se = cudaStreamSynchronize(nullptr);
    if (se != cudaSuccess) {
      fprintf(stderr, "[wtlgpu] sync before capture failed: %s\n", cudaGetErrorString(se));
      cudaGetLastError();
      return -10;
    }
    c->call_stream = c->graph_stream;
  }
  cudaError_t e = cudaStreamBeginCapture(c->call_stream, cudaStreamCaptureModeThreadLocal);
  if (e != cudaSuccess) {
    fprintf(stderr, "[wtlgpu] cudaStreamBeginCapture failed: %s\n", cudaGetErrorString(e));
    cudaGetLastError();
    return -10;
  }
  c->capturing = 1;
  return 0;
}

int wp_capture_end(WpCtx* c, void** out_exec) {
  if (!c || !out_exec) return -1;
  *out_exec = nullptr;
  if (!c->capturing) { fprintf(stderr, "[wtlgpu] capture_end without begin\n"); return -12; }
  cudaStream_t s = c->call_stream;
  c->capturing = 0;

  cudaGraph_t g = nullptr;
  cudaError_t e = cudaStreamEndCapture(s, &g);
  if (e != cudaSuccess) {
    /* Capture was invalidated by an error inside the region.  Clear the sticky error
     * and make absolutely sure the stream is no longer in capture mode, so the caller's
     * process is not left with a broken (capturing) stream. */
    cudaGetLastError();
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (s) cudaStreamIsCapturing(s, &cs);
    if (cs != cudaStreamCaptureStatusNone && s) {
      cudaGraph_t g2 = nullptr;
      cudaStreamEndCapture(s, &g2);
      if (g2) cudaGraphDestroy(g2);
      cudaGetLastError();
    }
    fprintf(stderr, "[wtlgpu] capture invalidated (%s); stream is clean, no graph created\n",
            cudaGetErrorString(e));
    return -13;
  }
  if (!g) return -14;
  size_t nodes = 0;
  cudaGraphGetNodes(g, nullptr, &nodes);
  c->last_graph_nodes = nodes;
  if (nodes == 0) {
    /* nothing was enqueued in the capture region (the caller almost certainly hit a
     * host-side error such as an un-warmed shape returning -6) -> do not hand back
     * a useless empty graph */
    cudaGraphDestroy(g);
    fprintf(stderr, "[wtlgpu] capture produced 0 nodes -> -14\n");
    return -14;
  }
  cudaGraphExec_t ex = nullptr;
  e = cudaGraphInstantiate(&ex, g, 0);
  cudaGraphDestroy(g);
  if (e != cudaSuccess) {
    fprintf(stderr, "[wtlgpu] cudaGraphInstantiate failed: %s\n", cudaGetErrorString(e));
    cudaGetLastError();
    return -15;
  }
  *out_exec = (void*)ex;
  return 0;
}

int wp_replay(WpCtx* c, void* exec) {
  if (!c || !exec) return -1;
  cudaError_t e = cudaGraphLaunch((cudaGraphExec_t)exec, c->call_stream);
  if (e != cudaSuccess) {
    fprintf(stderr, "[wtlgpu] cudaGraphLaunch failed: %s\n", cudaGetErrorString(e));
    cudaGetLastError();
    return -17;
  }
  return 0;
}

int wp_graph_free(WpCtx* c, void* exec) {
  (void)c;
  if (!exec) return 0;
  cudaError_t e = cudaGraphExecDestroy((cudaGraphExec_t)exec);
  if (e != cudaSuccess) { cudaGetLastError(); return -18; }
  return 0;
}

size_t wp_graph_count(WpCtx* c) { return c ? c->last_graph_nodes : 0; }
int wp_is_capturing(WpCtx* c) { return c ? c->capturing : 0; }

} /* extern "C" */
