/* wtlgpu.h -- minimal C ABI for the wtl int8 GPU GEMM primitive (sm_120 / RTX 5060 Ti).
 *
 * Layout contract (all matrices row-major, as in the Rust CPU reference):
 *   w    : [n][k]  int8   at device offset w_off inside the weight blob
 *   xq   : [t][k]  int8   device buffer
 *   xs   : [t]     float  per-row (per output frame) activation scale, device buffer
 *   bias : [n]     float  device buffer (wp_gemm2/wp_gemm_q/wp_gemm_qg), or a blob offset (wp_gemm)
 *   out  : [t][n]  float  device buffer
 *
 *   out[i][j] = ( (int) sum_k xq[i][k]*w[j][k] ) * xs[i] * ws + (bias ? bias[j] : 0)
 *
 * Streams: wp_gemm() and wp_gemm2(..., stream=NULL, ...) run on the *legacy default
 * stream* (stream 0).  For CUDA-graph capture call wp_capture_begin(): that switches
 * the context's stream to a private non-blocking stream (legacy 0 cannot be captured),
 * after which wp_stream(c) returns that same stream -- pass it to your own kernels and
 * to wp_gemm2() so everything lands in one graph.  wp_set_stream(c, NULL) restores 0.
 *
 * Shape warmup: the first call for a given (k,n,t) performs a cuBLASLt algorithm
 * search (it launches kernels), so it must happen OUTSIDE a capture; inside a capture
 * an un-warmed shape returns -6.
 */
#ifndef WTLGPU_H
#define WTLGPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WpCtx WpCtx;

/* Upload the whole weight blob (all layers) to the GPU once. */
WpCtx* wp_create(const void* w_host, size_t bytes);
void   wp_destroy(WpCtx* c);

void*  wp_alloc(size_t bytes);
void   wp_free(void* p);

/* ---- v1 ABI (unchanged) ------------------------------------------------- */
/* out[t][n] f32 = (xq[t][k] . w[n][k]) * xs[i] * ws + bias[j]
 * bias_off = byte offset of the f32 [n] bias inside the weight blob;
 *            pass (size_t)-1 for "no bias".  Runs on the context's call stream. */
int    wp_gemm(WpCtx* c, const int8_t* xq, const float* xs, int t, int k,
               size_t w_off, int n, float ws, size_t bias_off, float* out);

/* ---- v2: device bias pointer + explicit stream --------------------------- */
/* bias_dev == NULL -> no bias.  stream == NULL -> legacy default stream (0).
 * Everything (cuBLASLt call + epilogue) is enqueued on `stream`. */
int    wp_gemm2(WpCtx* c, const int8_t* xq, const float* xs, int t, int k,
                size_t w_off, int n, float ws, const float* bias_dev, void* stream,
                float* out);

/* ---- fused scale+bias+per-row dynamic quant (optional fast path) -------- */
/* Same GEMM, but instead of f32 out it emits the int8 re-quantisation:
 *    y[i][j] = acc*xs[i]*ws + bias[j]
 *    s[i]    = (max_j |y[i][j]| > 0) ? max/127 : 1.0        (same as Rust `quant`)
 *    out_q[i][j] = round(y[i][j] / s[i]), clamped to [-127,127]
 * Feed straight into the next wp_gemm as (xq=out_q, xs=out_s).
 * out_q: [t][n] int8, out_s: [t] f32. */
int    wp_gemm_q(WpCtx* c, const int8_t* xq, const float* xs, int t, int k,
                 size_t w_off, int n, float ws, const float* bias_dev, void* stream,
                 int8_t* out_q, float* out_s);

/* ---- fused scale+bias+GELU+per-row dynamic quant ----------------------- */
/*        y  = acc*xs[i]*ws + bias[j]
 *        g  = 0.5*y*(1+tanh(0.7978846*(y+0.044715*y^3)))     (same as Rust `gelu`)
 *        s  = (max|g|>0) ? max/127 : 1.0 ;  out_q = round(g/s) clamped
 * Replaces "FF-up GEMM + GELU kernel + quant kernel" with one kernel. */
int    wp_gemm_qg(WpCtx* c, const int8_t* xq, const float* xs, int t, int k,
                  size_t w_off, int n, float ws, const float* bias_dev, void* stream,
                  int8_t* out_q, float* out_s);

/* ---- standalone per-row dynamic quant (no GEMM) ------------------------ */
/* same rounding as the fused epilogue / Rust `quant()`.
 * Requires n % 256 == 0 && n/256 <= 16 (512/2048/4096 ok). */
int    wp_quant_rows(const float* y, int t, int n, void* stream, int8_t* out_q, float* out_s);

/* ---- CUDA graph: capture the whole window (40 layers, ~600 kernels) ---- */
/* wp_capture_begin: syncs the legacy stream, switches the context to the private
 *   graph stream (see wp_stream) and starts ThreadLocal capture on it.
 *   Returns 0, or -10 (already capturing / capture start failed).
 * wp_capture_end: ends capture, counts nodes, instantiates; *out_exec receives the
 *   graph-exec handle (NULL on failure).  On ANY error the stream is guaranteed to
 *   be out of capture mode before returning (returns -13, nothing left dangling).
 * wp_replay: cudaGraphLaunch(exec, current stream).
 * wp_graph_free: cudaGraphExecDestroy.
 * wp_graph_count: node count of the most recently captured graph (for logging). */
int    wp_capture_begin(WpCtx* c);
int    wp_capture_end(WpCtx* c, void** out_exec);
int    wp_replay(WpCtx* c, void* exec);
int    wp_graph_free(WpCtx* c, void* exec);
size_t wp_graph_count(WpCtx* c);
int    wp_is_capturing(WpCtx* c);

/* ---- additive helpers -------------------------------------------------- */
int    wp_h2d(void* dev, const void* host, size_t bytes);   /* sync H->D */
int    wp_d2h(void* host, const void* dev, size_t bytes);   /* sync D->H */
void*  wp_devptr(WpCtx* c, size_t off);                     /* device ptr into blob */
int    wp_sync(WpCtx* c);                                   /* sync current call stream */
int    wp_set_stream(WpCtx* c, void* stream);               /* default call stream (NULL=0) */
void*  wp_stream(WpCtx* c);                                 /* current call stream */
size_t wp_scratch_bytes(WpCtx* c);                          /* int32 acc scratch size */
const char* wp_version(void);

#ifdef __cplusplus
}
#endif
#endif /* WTLGPU_H */
