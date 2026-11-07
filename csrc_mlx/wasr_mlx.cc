// wasr_mlx.cc —— Apple Silicon / MLX 计算后端（对应 CUDA 版的 csrc/*.cu）
//
// 设计目标：和 CUDA 后端**同一个形状**——Rust 侧负责 fbank / .xnet 解析 / 权重打包 / 分块解码 /
// CTC / CLI，这里只做"一个窗口的前向"，输入 [t][40] 特征帧、输出 t1 个 token id。
//
// 与 CUDA 版的关键差异（数值路径，不是结构）：
//   CUDA: int8 权重 × per-row 动态 int8 激活，int32 累加，融合反量化 epilogue
//   这里: 权重在 create 时一次性反量化成 fp16；前向全 fp16（matmul 内部 fp32 累加），
//         attention 走 MLX 的 flash-attention 原语，LN/GELU 在 fp32 里算再回 fp16。
//   原因：Apple GPU（M1–M4）没有 int8 矩阵单元，MLX 的 matmul 也不收整数。
//   实测（scripts/mlx/bench_num.py）：fp16 路线的相对误差 4.6e-4，比 int8 激活量化本身的
//   4e-3 还小一个数量级；与 CUDA 版转录对比 CER 1.2%。
//
// ABI：见下方 extern "C"。Rust 侧 src/mlx.rs 一一对应。

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <mlx/compile.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <utility>
#include <vector>

using namespace mlx::core;

// WASR_MLX_DEBUG=1 时打点，方便定位崩溃/卡住的位置
static const bool WL_DBG = std::getenv("WASR_MLX_DEBUG") != nullptr;
#define WL_LOG(...)                         \
  do {                                      \
    if (WL_DBG) {                           \
      std::fprintf(stderr, __VA_ARGS__);    \
      std::fflush(stderr);                  \
    }                                       \
  } while (0)

// ---------------------------------------------------------------------------
// 与 Rust 共享的 plan 结构（#[repr(C)] 对应）
// ---------------------------------------------------------------------------
typedef struct {
  size_t pn_g;
  size_t pn_b;
  size_t mn_g;
  size_t mn_b;
  // 字段顺序必须与 Rust 侧 src/mlx.rs 的 WlLayerPlan 逐字段一致（偏移会经过 padding 校验）
  size_t q;
  float qs;
  size_t k;
  float ks;
  size_t v;
  float vs;
  size_t o;
  float os;
  size_t f1;
  float f1s;
  size_t f1b;
  size_t f2;
  float f2s;
  size_t f2b;
} wl_layer_plan;

typedef struct {
  const wl_layer_plan* layers;
  int n_layers;
  size_t lin;
  float lin_s;
  size_t lin_b;
  size_t out;                        // 输出层 int8 权重（词表已补齐到 4 的倍数）
  float out_s;
  size_t out_b;                      // f32 bias，尾部用 -1e30 压死 padding 列
  int vocab;                         // = out_n（补齐后的列数）
  int d, ff, flat, heads;
  size_t c1, c2, c3;                 // 卷积权重 f32，xnet 布局 [O][I][KH][KW]
} wl_plan;

// ---------------------------------------------------------------------------
// 权重容器
// ---------------------------------------------------------------------------
struct LayerW {
  array pn_g, pn_b, mn_g, mn_b;      // (D,) f32
  array wq, wk, wv, wo;              // (D, D)  f16，已转成 (k, n)，前向直接 y @ W
  array wf1;                         // (D, FF) f16
  array f1b;                         // (FF,)   f32
  array wf2;                         // (FF, D) f16
  array f2b;                         // (D,)    f32
  // MLX 的 array 没有默认构造，这里用 1 元素占位，create 时整体覆盖
  LayerW()
      : pn_g(zeros({1}, float32)), pn_b(zeros({1}, float32)),
        mn_g(zeros({1}, float32)), mn_b(zeros({1}, float32)),
        wq(zeros({1}, float16)), wk(zeros({1}, float16)), wv(zeros({1}, float16)),
        wo(zeros({1}, float16)), wf1(zeros({1}, float16)), f1b(zeros({1}, float32)),
        wf2(zeros({1}, float16)), f2b(zeros({1}, float32)) {}
};

struct WlCtx {
  std::vector<uint8_t> blob;         // 权重 blob 的副本（反量化时的原始视图）
  int D = 512, FF = 2048, FLAT = 3840, HEADS = 8, HD = 64, VOCAB = 0, NLAYER = 40;
  array c1, c2, c3;                  // (O,KH,KW,I) f16
  std::vector<LayerW> layers;
  array lin, lin_b, wout, out_b;
  // 按 (B, t1) 缓存编译后的整窗图（等价于 CUDA graph 的 per-t1 缓存）
  std::map<std::pair<int, int>, std::function<std::vector<array>(const std::vector<array>&)>> graphs;
  WlCtx()
      : c1(zeros({1}, float16)), c2(zeros({1}, float16)), c3(zeros({1}, float16)),
        lin(zeros({1}, float16)), lin_b(zeros({1}, float32)),
        wout(zeros({1}, float16)), out_b(zeros({1}, float32)) {}
};

// ---------------------------------------------------------------------------
// 权重反量化 / 布局转换
// ---------------------------------------------------------------------------
static array deq_mat(const uint8_t* base, size_t off, int n, int k, float s) {
  const int8_t* p = reinterpret_cast<const int8_t*>(base + off);
  std::vector<float> tmp(static_cast<size_t>(n) * static_cast<size_t>(k));
  for (size_t i = 0; i < tmp.size(); ++i) {
    tmp[i] = static_cast<float>(p[i]) * s;
  }
  array w(tmp.data(), {n, k}, float32);
  return astype(transpose(w, {1, 0}), float16);  // (n,k) → (k,n)
}

static array f32_vec(const uint8_t* base, size_t off, int n) {
  const float* p = reinterpret_cast<const float*>(base + off);
  return astype(array(p, {n}, float32), float32);
}

static array conv_w(const uint8_t* base, size_t off, int o, int i, int kh, int kw) {
  const float* p = reinterpret_cast<const float*>(base + off);
  array w(p, {o, i, kh, kw}, float32);
  return astype(transpose(w, {0, 2, 3, 1}), float16);  // (O,I,KH,KW) → (O,KH,KW,I)
}

// ---------------------------------------------------------------------------
// 小算子（口径与 src/model.rs 一致）
// ---------------------------------------------------------------------------
static array relu(const array& x) {
  return maximum(x, array(0.0f, float16));
}

static array var_(const array& a, int axis, bool keepdims) {
  return var(a, axis, keepdims, /*ddof=*/0);
}

/// LayerNorm：eps=1e-6，fp32 内部算，回 fp16（对齐 Rust 的 f32 累加）
static array layernorm(const array& x, const array& g, const array& b) {
  array xf = astype(x, float32);
  array mu = mean(xf, -1, true);
  array d = subtract(xf, mu);
  array v = var_(d, -1, true);
  array y = multiply(multiply(d, rsqrt(add(v, array(1e-6f)))), g);
  return astype(add(y, b), float16);
}

/// tanh 近似 GELU：0.5x(1+tanh(0.7978846(x+0.044715x³)))，与 Rust 的 gelu() 同式
static array gelu(const array& x) {
  array xf = astype(x, float32);
  array inner = multiply(
      array(0.7978846f),
      add(xf, multiply(array(0.044715f), multiply(xf, square(xf)))));
  return astype(multiply(multiply(array(0.5f), xf), add(array(1.0f), tanh(inner))), float16);
}

// ---------------------------------------------------------------------------
// 卷积前端：[B][t][40] → [B*t1][FLAT]
//   conv1 5×5 1→32，时频各补 2
//   conv2 7×7 32→64，时间 stride 5 补 1；频率**左补 3 右补 4**（输出 41 列，与 Rust 一致）
//   conv3 3×3 64→192，频率 stride 2，时间补 1
// ---------------------------------------------------------------------------
static array frontend(WlCtx* c, const array& x, int B, int t1) {
  array y = relu(conv2d(pad(x, std::vector<std::pair<int, int>>{{0, 0}, {2, 2}, {2, 2}, {0, 0}}), c->c1));
  y = relu(conv2d(pad(y, std::vector<std::pair<int, int>>{{0, 0}, {1, 1}, {3, 4}, {0, 0}}),
                  c->c2, std::pair<int, int>{5, 1}));
  y = relu(conv2d(pad(y, std::vector<std::pair<int, int>>{{0, 0}, {1, 1}, {0, 0}, {0, 0}}),
                  c->c3, std::pair<int, int>{1, 2}));
  // (B,t1,20,192) → (B,t1,192,20) → (B*t1, 3840)，对应 Rust 的 [o*20 + f] 展开顺序
  return reshape(transpose(y, {0, 1, 3, 2}), {B * t1, c->FLAT});
}

// ---------------------------------------------------------------------------
// 整窗前向：卷积前端 + linear + 40 层 + 输出层 + argmax
// ---------------------------------------------------------------------------
static array window_ops(WlCtx* c, const array& x, int B, int t1) {
  WL_LOG("[wl]   frontend (B=%d t1=%d)...\n", B, t1);
  array h = add(matmul(frontend(c, x, B, t1), c->lin), c->lin_b);
  WL_LOG("[wl]   linear ok %d x %d\n", (int)h.shape(0), (int)h.shape(1));
  const int D = c->D, H = c->HEADS, HD = c->HD;
  for (int li = 0; li < (int)c->layers.size(); ++li) {
    const auto& l = c->layers[(size_t)li];
    if (WL_DBG && li == 0) {
      WL_LOG("[wl]   layer 0 starting...\n");
    }
    array y = layernorm(h, l.pn_g, l.pn_b);
    array q = transpose(reshape(matmul(y, l.wq), {B, t1, H, HD}), {0, 2, 1, 3});
    array k = transpose(reshape(matmul(y, l.wk), {B, t1, H, HD}), {0, 2, 1, 3});
    array v = transpose(reshape(matmul(y, l.wv), {B, t1, H, HD}), {0, 2, 1, 3});
    if (WL_DBG && li == 0) {
      WL_LOG("[wl]   q/k/v ok\n");
    }
    array a = fast::scaled_dot_product_attention(q, k, v, 0.125f);
    if (WL_DBG && li == 0) {
      WL_LOG("[wl]   sdpa ok %d %d %d %d\n", (int)a.shape(0), (int)a.shape(1), (int)a.shape(2), (int)a.shape(3));
    }
    h = add(h, matmul(reshape(transpose(a, {0, 2, 1, 3}), {B * t1, D}), l.wo));

    array y2 = layernorm(h, l.mn_g, l.mn_b);
    array f = gelu(add(matmul(y2, l.wf1), l.f1b));
    h = add(h, add(matmul(f, l.wf2), l.f2b));
    if (WL_DBG && li == 0) {
      WL_LOG("[wl]   layer 0 done\n");
    }
  }
  array lo = add(matmul(h, c->wout), c->out_b);
  WL_LOG("[wl]   out %d x %d, argmax...\n", (int)lo.shape(0), (int)lo.shape(1));
  return argmax(lo, -1);  // (B*t1,) uint32
}

static int run(WlCtx* c, const float* feats, int B, int t, uint32_t* ids) {
  if (t < 5) {
    return 0;
  }
  const int t1 = (t - 5) / 5 + 1;
  WL_LOG("[wl] run B=%d t=%d t1=%d\n", B, t, t1);
  // feats: (B, t, 40) f32 → (B, t, 40, 1) f16
  array x = astype(array(feats, {B, t, 40}, float32), float16);
  x = reshape(x, {B, t, 40, 1});
  auto key = std::make_pair(B, t1);
  auto it = c->graphs.find(key);
  if (it == c->graphs.end()) {
    WL_LOG("[wl] compiling graph for B=%d t1=%d\n", B, t1);
    // 首次遇到该形状：编译整窗图（等价于 CUDA 的 capture）
    auto fn = [c, B, t1](const std::vector<array>& in) -> std::vector<array> {
      return {window_ops(c, in[0], B, t1)};
    };
    auto compiled = compile(std::function<std::vector<array>(const std::vector<array>&)>(fn));
    it = c->graphs.emplace(key, compiled).first;
    WL_LOG("[wl] compiled\n");
  }
  array out = it->second({x})[0];
  WL_LOG("[wl] eval...\n");
  eval(out);
  WL_LOG("[wl] eval done\n");
  const size_t n = static_cast<size_t>(B) * static_cast<size_t>(t1);
  std::memcpy(ids, out.data<uint32_t>(), n * sizeof(uint32_t));
  return t1;
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------
extern "C" {

WlCtx* wl_create(const void* blob, size_t bytes, const wl_plan* plan) {
  WL_LOG("[wl] create: blob=%zu bytes n_layers=%d D=%d FF=%d FLAT=%d V=%d\n", bytes,
         plan ? plan->n_layers : -1, plan ? plan->d : -1, plan ? plan->ff : -1,
         plan ? plan->flat : -1, plan ? plan->vocab : -1);
  if (blob == nullptr || plan == nullptr || plan->n_layers <= 0) {
    return nullptr;
  }
  auto* c = new WlCtx();
  c->blob.assign(reinterpret_cast<const uint8_t*>(blob),
                 reinterpret_cast<const uint8_t*>(blob) + bytes);
  const uint8_t* base = c->blob.data();
  c->D = plan->d;
  c->FF = plan->ff;
  c->FLAT = plan->flat;
  c->HEADS = plan->heads;
  c->HD = plan->d / plan->heads;
  c->VOCAB = plan->vocab;
  c->NLAYER = plan->n_layers;

  c->c1 = conv_w(base, plan->c1, 32, 1, 5, 5);
  WL_LOG("[wl] conv1 ok\n");
  c->c2 = conv_w(base, plan->c2, 64, 32, 7, 7);
  c->c3 = conv_w(base, plan->c3, 192, 64, 3, 3);
  WL_LOG("[wl] conv ok\n");

  c->layers.resize(static_cast<size_t>(plan->n_layers));
  for (int i = 0; i < plan->n_layers; ++i) {
    const wl_layer_plan& p = plan->layers[i];
    LayerW& l = c->layers[static_cast<size_t>(i)];
    l.pn_g = f32_vec(base, p.pn_g, c->D);
    l.pn_b = f32_vec(base, p.pn_b, c->D);
    l.mn_g = f32_vec(base, p.mn_g, c->D);
    l.mn_b = f32_vec(base, p.mn_b, c->D);
    l.wq = deq_mat(base, p.q, c->D, c->D, p.qs);
    l.wk = deq_mat(base, p.k, c->D, c->D, p.ks);
    l.wv = deq_mat(base, p.v, c->D, c->D, p.vs);
    l.wo = deq_mat(base, p.o, c->D, c->D, p.os);
    l.wf1 = deq_mat(base, p.f1, c->FF, c->D, p.f1s);
    l.f1b = f32_vec(base, p.f1b, c->FF);
    l.wf2 = deq_mat(base, p.f2, c->D, c->FF, p.f2s);
    l.f2b = f32_vec(base, p.f2b, c->D);
    if (WL_DBG && i % 10 == 0) {
      WL_LOG("[wl] layer %d ok\n", i);
    }
  }
  c->lin = deq_mat(base, plan->lin, c->D, c->FLAT, plan->lin_s);
  c->lin_b = f32_vec(base, plan->lin_b, c->D);
  c->wout = deq_mat(base, plan->out, c->VOCAB, c->D, plan->out_s);
  c->out_b = f32_vec(base, plan->out_b, c->VOCAB);
  WL_LOG("[wl] tensors built, evaluating...\n");
  eval(c->c1, c->c2, c->c3, c->lin, c->lin_b, c->wout, c->out_b);
  for (auto& l : c->layers) {
    eval(l.pn_g, l.pn_b, l.mn_g, l.mn_b, l.wq, l.wk, l.wv, l.wo, l.wf1, l.f1b, l.wf2, l.f2b);
  }
  return c;
}

void wl_destroy(WlCtx* c) {
  delete c;
}

/// 单窗前向：feats[t][40] f32 → ids[t1]；返回 t1，失败 -1
int wl_forward(WlCtx* c, const float* feats, int t, uint32_t* ids) {
  try {
    return run(c, feats, 1, t, ids);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[wl] wl_forward 异常: %s\n", e.what());
    return -1;
  } catch (...) {
    std::fprintf(stderr, "[wl] wl_forward 未知异常\n");
    return -1;
  }
}

/// 批窗前向：feats[B][t][40] → ids[B][t1]；返回 t1，失败 -1
int wl_forward_batch(WlCtx* c, const float* feats, int b, int t, uint32_t* ids) {
  try {
    return run(c, feats, b, t, ids);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[wl] wl_forward_batch 异常: %s\n", e.what());
    return -1;
  } catch (...) {
    std::fprintf(stderr, "[wl] wl_forward_batch 未知异常\n");
    return -1;
  }
}

int wl_vocab(WlCtx* c) {
  return c->VOCAB;
}

/// ABI 自检：Rust 侧拿这两个 size 和自己的 repr(C) 结构比，防止字段顺序/填充漂移再次引发段错误
void wl_abi_sizes(int* layer_size, int* plan_size) {
  if (layer_size) {
    *layer_size = static_cast<int>(sizeof(wl_layer_plan));
  }
  if (plan_size) {
    *plan_size = static_cast<int>(sizeof(wl_plan));
  }
}

}  // extern "C"
