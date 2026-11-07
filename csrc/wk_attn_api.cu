/* wk_attn_api.cu -- single C ABI front-end for the attention kernels.
 *
 *   int wk_attention_fast(void* stream, const float* q, const float* k, const float* v,
 *                         int t, int d, int heads, float* scratch, float* out);
 *
 * q/k/v/out : [t][d] row-major f32,  heads = 8, head_dim = 64  (d = 512)
 * scratch   : device buffer of at least heads*t*t floats (only the cuBLAS path uses it,
 *             the tiled path ignores it -> may be NULL there)
 * stream    : CUDA stream (NULL = legacy default stream); calls are async.
 * returns 0 on success, negative on error.
 *
 * Default implementation = the faster one measured on RTX 5060 Ti.
 * Override with env WTL_ATTN_IMPL=cublas|tiled for A/B testing.
 */
#include <cstdlib>
#include <cstring>

extern "C" int wk_attention_cublas(void* stream, const float* q, const float* k, const float* v,
                                   int t, int d, int heads, float* scratch, float* out);
extern "C" int wk_attention_tiled(void* stream, const float* q, const float* k, const float* v,
                                  int t, int d, int heads, float* scratch, float* out);

extern "C" int wk_attention_fast(void* stream, const float* q, const float* k, const float* v,
                                 int t, int d, int heads, float* scratch, float* out) {
  const char* e = getenv("WTL_ATTN_IMPL");
  if (e && e[0] == 't') return wk_attention_tiled(stream, q, k, v, t, d, heads, scratch, out);
  return wk_attention_cublas(stream, q, k, v, t, d, heads, scratch, out);
}

extern "C" const char* wk_attention_version(void) {
  return "wk_attention 1.1 (sm_120; cuBLAS 2xSgemm+softmax and tiled fp32; hd=64)";
}
