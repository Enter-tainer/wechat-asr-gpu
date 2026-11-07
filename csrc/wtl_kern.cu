//! wtlkern：wtl 引擎的 GPU 算子层。
//!
//! 分工：
//!   * 矩阵乘        → libwtlgpu.so 的 cublasLt int8（不在本文件）
//!   * attention     → cuBLAS 的 batched sgemm（q·kᵀ、P·V）+ 一个 softmax kernel
//!   * 卷积前端      → 3 个直接卷积 kernel（conv1/conv2/conv3，与 model.rs 的 CPU 版同语义）
//!   * norm+quant / add+norm+quant / gelu+quant / quant / argmax → 手写融合 kernel
//!
//! 口径对齐 wtl-linux/src/model.rs。归一化/量化的浮点运算顺序尽量与 CPU 一致（可用对拍），
//! attention 的 softmax 归约和卷积累加用并行归约（浮点末位可能与 CPU 不同，量级 1e-7）。
//!
//! 所有函数接受 `void* stream`（传 libwtlgpu 的 wp_stream(ctx)），与 GEMM 同流。
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CK(x)                                                                     \
    do {                                                                          \
        cudaError_t e_ = (x);                                                     \
        if (e_ != cudaSuccess) {                                                  \
            fprintf(stderr, "wtlkern CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            abort();                                                              \
        }                                                                         \
    } while (0)
#define CBLK(x)                                                                   \
    do {                                                                          \
        cublasStatus_t e_ = (x);                                                  \
        if (e_ != CUBLAS_STATUS_SUCCESS) {                                        \
            fprintf(stderr, "wtlkern cuBLAS error %d at %s:%d\n", (int)e_, __FILE__, __LINE__); \
            abort();                                                              \
        }                                                                         \
    } while (0)

// 模型结构常量（与 src/model.rs 保持一致）
#define DIM_ 40
#define F1_ 32
#define F2_ 64
#define F3_ 192
#define W2_ (DIM_ + 1)   // 41，第 41 列恒 0（频率维补边）
#define K2_ (F1_ * 49)   // 1568
#define K3_ (F2_ * 9)    // 576

// ---------------------------------------------------------------- 卷积前端
// conv1: [t][40] → [t][32][40]，5×5，时频各补 2
__global__ void conv1_kernel(const float* __restrict__ x, const float* __restrict__ w, int t,
                             float* __restrict__ y1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = t * F1_ * DIM_;
    if (idx >= total) return;
    int f = idx % DIM_, o = (idx / DIM_) % F1_, ti = idx / (DIM_ * F1_);
    float acc = 0.f;
    for (int i = 0; i < 5; i++) {
        int tt = ti + i - 2;
        if (tt < 0 || tt >= t) continue;
        for (int j = 0; j < 5; j++) {
            int ff = f + j - 2;
            if (ff < 0 || ff >= DIM_) continue;
            acc += w[o * 25 + i * 5 + j] * x[(size_t)tt * DIM_ + ff];
        }
    }
    y1[idx] = acc > 0.f ? acc : 0.f;
}

// conv2: [t][32][40] -> [t1][64][41]，7×7，时补 1 步长 5、频补 3，第 41 列填 0
__global__ void conv2_kernel(const float* __restrict__ y1, const float* __restrict__ w, int t, int t1,
                             float* __restrict__ y2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = t1 * F2_ * W2_;
    if (idx >= total) return;
    int f = idx % W2_, o = (idx / W2_) % F2_, ti = idx / (W2_ * F2_);
    if (f == DIM_) { y2[idx] = 0.f; return; }
    float a0 = 0.f, a1 = 0.f;
    const float* wp = w + (size_t)o * K2_;
    for (int i = 0; i < 7; i++) {
        int tt = ti * 5 + i - 1;
        if (tt < 0 || tt >= t) continue;
        const float* row = y1 + (size_t)tt * F1_ * DIM_;
        for (int c = 0; c < F1_; c++) {
            const float* wr = wp + c * 49 + i * 7;
            const float* xr = row + c * DIM_;
            for (int j = 0; j < 7; j++) {
                int ff = f + j - 3;
                if (ff < 0 || ff >= DIM_) continue;
                float p = wr[j] * xr[ff];
                if (j & 1) a1 += p; else a0 += p;
            }
        }
    }
    float acc = a0 + a1;
    y2[idx] = acc > 0.f ? acc : 0.f;
}

// conv3: [t1][64][41] -> [t1][192][20]，3×3，时补 1、频率步长 2
__global__ void conv3_kernel(const float* __restrict__ y2, const float* __restrict__ w, int t1,
                             float* __restrict__ y3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = t1 * F3_ * 20;
    if (idx >= total) return;
    int f = idx % 20, o = (idx / 20) % F3_, ti = idx / (20 * F3_);
    float a0 = 0.f, a1 = 0.f;
    const float* wp = w + (size_t)o * K3_;
    for (int i = 0; i < 3; i++) {
        int tt = ti + i - 1;
        if (tt < 0 || tt >= t1) continue;
        const float* row = y2 + (size_t)tt * F2_ * W2_;
        for (int c = 0; c < F2_; c++) {
            const float* wr = wp + c * 9 + i * 3;
            const float* xr = row + c * W2_ + 2 * f;
            for (int j = 0; j < 3; j++) {
                float p = wr[j] * xr[j];
                if (j & 1) a1 += p; else a0 += p;
            }
        }
    }
    float acc = a0 + a1;
    y3[idx] = acc > 0.f ? acc : 0.f;
}

// ---------------------------------------------------------------- norm / quant
// 一行一个 block。均值/方差在 thread0 顺序累加（同 CPU），归一化与量化并行。
__global__ void norm_quant_kernel(const float* __restrict__ h, const float* __restrict__ g,
                                  const float* __restrict__ b, int t, int d,
                                  int8_t* __restrict__ xq, float* __restrict__ xs) {
    const int row = blockIdx.x;
    if (row >= t) return;
    __shared__ float red[256];
    __shared__ float s_mean, s_inv;
    const float* x = h + (size_t)row * d;
    const int tid = threadIdx.x;

    if (tid == 0) {
        float sum = 0.f;
        for (int i = 0; i < d; i++) sum += x[i];
        float mean = sum / d;
        float s2 = 0.f;
        for (int i = 0; i < d; i++) { float dd = x[i] - mean; s2 += dd * dd; }
        s_mean = mean;
        s_inv = 1.f / sqrtf(s2 / d + 1e-6f);
    }
    __syncthreads();
    const float mean = s_mean, inv = s_inv;

    float mx = 0.f;
    for (int i = tid; i < d; i += blockDim.x) {
        float y = __fadd_rn(__fmul_rn(__fmul_rn(x[i] - mean, inv), g[i]), b[i]);
        mx = fmaxf(mx, fabsf(y));
    }
    red[tid] = mx;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    float m = red[0];
    float sc = m > 0.f ? m / 127.f : 1.f;
    float inv_s = 1.f / sc;
    if (tid == 0) xs[row] = sc;
    for (int i = tid; i < d; i += blockDim.x) {
        float y = __fadd_rn(__fmul_rn(__fmul_rn(x[i] - mean, inv), g[i]), b[i]);
        float q = roundf(y * inv_s);
        q = q > 127.f ? 127.f : (q < -127.f ? -127.f : q);
        xq[(size_t)row * d + i] = (int8_t)q;
    }
}

// h += o，然后对 h 做 norm+quant（残差 + 下一层 norm 融合）
__global__ void add_norm_quant_kernel(float* __restrict__ h, const float* __restrict__ o,
                                      const float* __restrict__ g, const float* __restrict__ b,
                                      int t, int d, int8_t* __restrict__ xq, float* __restrict__ xs) {
    const int row = blockIdx.x;
    if (row >= t) return;
    __shared__ float red[256];
    __shared__ float s_mean, s_inv;
    float* x = h + (size_t)row * d;
    const float* oo = o + (size_t)row * d;
    const int tid = threadIdx.x;
    for (int i = tid; i < d; i += blockDim.x) x[i] += oo[i];
    __syncthreads();

    if (tid == 0) {
        float sum = 0.f;
        for (int i = 0; i < d; i++) sum += x[i];
        float mean = sum / d;
        float s2 = 0.f;
        for (int i = 0; i < d; i++) { float dd = x[i] - mean; s2 += dd * dd; }
        s_mean = mean;
        s_inv = 1.f / sqrtf(s2 / d + 1e-6f);
    }
    __syncthreads();
    const float mean = s_mean, inv = s_inv;

    float mx = 0.f;
    for (int i = tid; i < d; i += blockDim.x) {
        float y = __fadd_rn(__fmul_rn(__fmul_rn(x[i] - mean, inv), g[i]), b[i]);
        mx = fmaxf(mx, fabsf(y));
    }
    red[tid] = mx;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    float m = red[0];
    float sc = m > 0.f ? m / 127.f : 1.f;
    float inv_s = 1.f / sc;
    if (tid == 0) xs[row] = sc;
    for (int i = tid; i < d; i += blockDim.x) {
        float y = __fadd_rn(__fmul_rn(__fmul_rn(x[i] - mean, inv), g[i]), b[i]);
        float q = roundf(y * inv_s);
        q = q > 127.f ? 127.f : (q < -127.f ? -127.f : q);
        xq[(size_t)row * d + i] = (int8_t)q;
    }
}

__global__ void quant_kernel(const float* __restrict__ x, int t, int k,
                             int8_t* __restrict__ xq, float* __restrict__ xs) {
    const int row = blockIdx.x;
    if (row >= t) return;
    __shared__ float red[256];
    const float* r = x + (size_t)row * k;
    const int tid = threadIdx.x;
    float mx = 0.f;
    for (int i = tid; i < k; i += blockDim.x) mx = fmaxf(mx, fabsf(r[i]));
    red[tid] = mx;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    float m = red[0];
    float sc = m > 0.f ? m / 127.f : 1.f;
    float inv_s = 1.f / sc;
    if (tid == 0) xs[row] = sc;
    for (int i = tid; i < k; i += blockDim.x) {
        float q = roundf(r[i] * inv_s);
        q = q > 127.f ? 127.f : (q < -127.f ? -127.f : q);
        xq[(size_t)row * k + i] = (int8_t)q;
    }
}

// gelu 原地 + 逐行 quant（一个 kernel 完成）
__global__ void gelu_quant_kernel(float* __restrict__ x, int t, int k,
                                  int8_t* __restrict__ xq, float* __restrict__ xs) {
    const int row = blockIdx.x;
    if (row >= t) return;
    __shared__ float red[256];
    float* r = x + (size_t)row * k;
    const int tid = threadIdx.x;
    for (int i = tid; i < k; i += blockDim.x) {
        float v = r[i];
        float u = __fmul_rn(0.7978846f, __fadd_rn(v, __fmul_rn(__fmul_rn(0.044715f, v), __fmul_rn(v, v))));
        r[i] = __fmul_rn(__fmul_rn(0.5f, v), __fadd_rn(1.f, tanhf(u)));
    }
    __syncthreads();
    float mx = 0.f;
    for (int i = tid; i < k; i += blockDim.x) mx = fmaxf(mx, fabsf(r[i]));
    red[tid] = mx;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    float m = red[0];
    float sc = m > 0.f ? m / 127.f : 1.f;
    float inv_s = 1.f / sc;
    if (tid == 0) xs[row] = sc;
    for (int i = tid; i < k; i += blockDim.x) {
        float q = roundf(r[i] * inv_s);
        q = q > 127.f ? 127.f : (q < -127.f ? -127.f : q);
        xq[(size_t)row * k + i] = (int8_t)q;
    }
}

__global__ void add_kernel(float* __restrict__ h, const float* __restrict__ a, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] += a[i];
}

__global__ void argmax_kernel(const float* __restrict__ x, int t, int n, uint32_t* __restrict__ out) {
    const int row = blockIdx.x;
    if (row >= t) return;
    const float* r = x + (size_t)row * n;
    __shared__ float bv[256];
    __shared__ int bi[256];
    const int tid = threadIdx.x;
    float best = -3.4e38f;
    int bj = 0;
    for (int j = tid; j < n; j += blockDim.x)
        if (r[j] > best) { best = r[j]; bj = j; }
    bv[tid] = best;
    bi[tid] = bj;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) {
            if (bv[tid + st] > bv[tid] || (bv[tid + st] == bv[tid] && bi[tid + st] < bi[tid])) {
                bv[tid] = bv[tid + st];
                bi[tid] = bi[tid + st];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[row] = (uint32_t)bi[0];
}

// ---------------------------------------------------------------- attention
// scores 由 cuBLAS batched sgemm 算出（列主序 [t][t]，第 h 个 batch 从 h*t*t 开始），
// 这个 kernel 只做 softmax（乘 0.125 之后）：max/exp/sum 均并行归约。
__global__ void softmax_kernel(float* __restrict__ s, int t, int heads, float scale) {
    const int pat = blockIdx.x;   // i*heads + h
    const int h = pat % heads;
    float* base = s + (size_t)h * t * t;
    __shared__ float red[128];
    const int tid = threadIdx.x;

    float mx = -3.4e38f;
    for (int j = tid; j < t; j += blockDim.x) mx = fmaxf(mx, base[(size_t)j * t + (pat / heads)] * scale);
    red[tid] = mx;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    const float m = red[0];
    __syncthreads();
    float z = 0.f;
    for (int j = tid; j < t; j += blockDim.x) {
        float e = expf(base[(size_t)j * t + (pat / heads)] * scale - m);
        base[(size_t)j * t + (pat / heads)] = e;
        z += e;
    }
    red[tid] = z;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    const float inv = 1.f / red[0];
    for (int j = tid; j < t; j += blockDim.x) base[(size_t)j * t + (pat / heads)] *= inv;
}

static cublasHandle_t g_h = nullptr;

// 融合 attention：一个 kernel 完成 QKᵀ + softmax + PV，不落 t×t 分数矩阵到显存。
// 每个 block 处理一个 (查询位置 i, head h)，threads = hd：
//   1) score[j] = Σ_x q[i][h*hd+x]*k[j][h*hd+x]（各线程分担 j，8 路累加器）
//   2) warp shuffle 做 max/exp/sum 归约（× 0.125）
//   3) out[d] = Σ_j (score[j]/z) * v[j][h*hd+d]（每线程一个 d）
__global__ void attention_fused_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                       const float* __restrict__ v, int t, int d, int heads,
                                       float* __restrict__ out) {
    extern __shared__ float sc[];   // t 个分数
    const int pat = blockIdx.x;     // i * heads + h
    const int i = pat / heads, h = pat % heads;
    const int hd = d / heads;
    const int lane = threadIdx.x;
    const float* qi = q + (size_t)i * d + h * hd;

    for (int j = lane; j < t; j += blockDim.x) {
        const float* kj = k + (size_t)j * d + h * hd;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
        int s = 0;
        for (; s + 8 <= hd; s += 8) {
            a0 += qi[s + 0] * kj[s + 0];
            a1 += qi[s + 1] * kj[s + 1];
            a2 += qi[s + 2] * kj[s + 2];
            a3 += qi[s + 3] * kj[s + 3];
            a4 += qi[s + 4] * kj[s + 4];
            a5 += qi[s + 5] * kj[s + 5];
            a6 += qi[s + 6] * kj[s + 6];
            a7 += qi[s + 7] * kj[s + 7];
        }
        float tot = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
        for (; s < hd; s++) tot += qi[s] * kj[s];
        sc[j] = tot * 0.125f;
    }
    __syncthreads();

    __shared__ float red[32];
    float lmax = -3.4e38f;
    for (int j = lane; j < t; j += blockDim.x) lmax = fmaxf(lmax, sc[j]);
    for (int off = 16; off > 0; off >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, off));
    if ((lane & 31) == 0) red[lane >> 5] = lmax;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) / 32;
    float m = red[0];
    for (int w = 1; w < nwarps; w++) m = fmaxf(m, red[w]);

    float lsum = 0.f;
    for (int j = lane; j < t; j += blockDim.x) {
        float e = expf(sc[j] - m);
        sc[j] = e;
        lsum += e;
    }
    for (int off = 16; off > 0; off >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
    if ((lane & 31) == 0) red[lane >> 5] = lsum;
    __syncthreads();
    float z = red[0];
    for (int w = 1; w < nwarps; w++) z += red[w];
    const float inv = 1.f / z;

    float acc = 0.f;
    for (int j = 0; j < t; j++) acc += sc[j] * inv * v[(size_t)j * d + h * hd + lane];
    out[(size_t)i * d + h * hd + lane] = acc;
}

extern "C" {

const char* wk_version(void) { return "wtlkern 2 (cublas sgemm attention + fused conv/quant)"; }

/// 卷积前端一次性跑完：x[t][40] → y1[t][32][40] → y2[t1][64][41] → y3[t1][192][20]
void wk_front(void* stream, const float* x, const float* c1, const float* c2, const float* c3,
              int t, int t1, float* y1, float* y2, float* y3) {
    cudaStream_t st = (cudaStream_t)stream;
    int n1 = t * F1_ * DIM_;
    conv1_kernel<<<(n1 + 255) / 256, 256, 0, st>>>(x, c1, t, y1);
    int n2 = t1 * F2_ * W2_;
    conv2_kernel<<<(n2 + 255) / 256, 256, 0, st>>>(y1, c2, t, t1, y2);
    int n3 = t1 * F3_ * 20;
    conv3_kernel<<<(n3 + 255) / 256, 256, 0, st>>>(y2, c3, t1, y3);
}

/// attention: q,k,v [t][d] 行主序（按 head 切分）→ out[t][d]；scratch 需 heads*t*t 个 float
void wk_attention(void* stream, const float* q, const float* k, const float* v, int t, int d,
                  int heads, float* scratch, float* out) {
    // 融合版（单 kernel）：实测比 cuBLAS(2×sgemm)+softmax 更慢，但数值正确；
    // cuBLAS 版的 correctness 还没查清（整模转录会乱），先保留融合版作为默认路径。
    (void)scratch;
    size_t shbytes = (size_t)t * sizeof(float);
    attention_fused_kernel<<<t * heads, d / heads, shbytes, (cudaStream_t)stream>>>(q, k, v, t, d, heads, out);
}

void wk_norm_quant(void* stream, const float* h, const float* g, const float* b, int t, int d,
                   int8_t* xq, float* xs) {
    norm_quant_kernel<<<t, 256, 0, (cudaStream_t)stream>>>(h, g, b, t, d, xq, xs);
}

void wk_add_norm_quant(void* stream, float* h, const float* o, const float* g, const float* b, int t,
                       int d, int8_t* xq, float* xs) {
    add_norm_quant_kernel<<<t, 256, 0, (cudaStream_t)stream>>>(h, o, g, b, t, d, xq, xs);
}

void wk_quant(void* stream, const float* x, int t, int k, int8_t* xq, float* xs) {
    quant_kernel<<<t, 256, 0, (cudaStream_t)stream>>>(x, t, k, xq, xs);
}

void wk_gelu_quant(void* stream, float* x, int t, int k, int8_t* xq, float* xs) {
    gelu_quant_kernel<<<t, 256, 0, (cudaStream_t)stream>>>(x, t, k, xq, xs);
}

void wk_add(void* stream, float* h, const float* a, int t, int d) {
    size_t n = (size_t)t * d;
    add_kernel<<<(int)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(h, a, n);
}

void wk_argmax(void* stream, const float* x, int t, int n, uint32_t* out) {
    argmax_kernel<<<t, 256, 0, (cudaStream_t)stream>>>(x, t, n, out);
}

int wk_last_error(void) { return (int)cudaGetLastError(); }

}  // extern "C"
