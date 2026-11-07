"""MLX 优化版实测：批处理窗口 + 融合 layernorm + 拆解。

关键想法：这个引擎是 CTC 编码器，**窗口之间没有模型状态依赖**（上下文只来自音频的
prev/post 重叠），所以批模式可以把 N 个窗口同时送进 GPU：
  - 所有 GEMM 变成 (B*t, k) @ (k, n) 的大 GEMM（利用率↑）
  - attention 用 batch 维 (B, H, t, dh) 的 SDPA
预期：单窗 104x 实时 → 批 4/8 窗时接近 fp16 峰值的一半。
"""
import sys
import time
import mlx.core as mx

D, FF, L, H, V = 512, 2048, 40, 8, 10504
DH = D // H
T = 1600
REPS = 3


def bench(fn, reps=REPS, warm=2):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / reps


def make_model(dt=mx.float16):
    scale = 0.02
    layers = []
    for _ in range(L):
        layers.append({
            "wqkv": (mx.random.normal((3 * D, D)) * scale).astype(dt),
            "wo": (mx.random.normal((D, D)) * scale).astype(dt),
            "w1": (mx.random.normal((FF, D)) * scale).astype(dt),
            "w2": (mx.random.normal((D, FF)) * scale).astype(dt),
            "g1": mx.ones((D,), dtype=dt), "b1": mx.zeros((D,), dtype=dt),
            "g2": mx.ones((D,), dtype=dt), "b2": mx.zeros((D,), dtype=dt),
        })
    wout = (mx.random.normal((V, D)) * scale).astype(dt)
    mx.eval([v for l in layers for v in l.values()] + [wout])
    return layers, wout


LN = None
try:
    mx.fast.layer_norm(mx.zeros((2, D), mx.float16), mx.ones((D,), mx.float16), mx.zeros((D,), mx.float16))
    LN = "fast"
except Exception:
    LN = "manual"


def layernorm(x, g, b, eps=1e-5):
    if LN == "fast":
        return mx.fast.layer_norm(x, g, b, eps)
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mu) ** 2, axis=-1, keepdims=True)
    return (x - mu) * mx.rsqrt(var + eps) * g + b


def forward(x, B, layers, wout, with_attn=True):
    """x: (B, T, D)"""
    BT = B * T
    h = x.reshape(BT, D)
    for p in layers:
        y = layernorm(h, p["g1"], p["b1"])
        qkv = (y @ p["wqkv"].T).reshape(B, T, 3, H, DH)
        q, k, v = mx.split(qkv, 3, axis=2)                  # (B,T,1,H,DH)
        q = q.reshape(B, T, H, DH).transpose(0, 2, 1, 3)    # (B,H,T,DH)
        k = k.reshape(B, T, H, DH).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, DH).transpose(0, 2, 1, 3)
        if with_attn:
            att = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / DH ** 0.5)
        else:
            att = q
        att = att.transpose(0, 2, 1, 3).reshape(BT, D)
        h = h + att @ p["wo"].T
        y = layernorm(h, p["g2"], p["b2"])
        f = mx.maximum(y @ p["w1"].T, 0)
        h = h + f @ p["w2"].T
    return h @ wout.T


mx.random.seed(0)
print(f"MLX {mx.__version__}  device={mx.default_device()}  layernorm={LN}  T={T}")
layers, wout = make_model()

rows = []
for B in (1, 2, 4, 8):
    x = (mx.random.normal((B, T, D)) * 0.1).astype(mx.float16)
    mx.eval(x)
    sec = bench(lambda B=B, x=x: forward(x, B, layers, wout))
    per_win = sec / B
    audio = T / 100.0
    rt = audio / per_win
    print(f"B={B:2d}: {sec*1e3:8.1f} ms/批  {per_win*1e3:7.1f} ms/窗  "
          f"RTF={per_win/audio:.4f}  {rt:6.1f}x 实时")
    rows.append((B, per_win, rt))

print("\n-- 拆解（B=4）：GEMM+LN+GELU 去掉 attention 后 --")
B = 4
x = (mx.random.normal((B, T, D)) * 0.1).astype(mx.float16)
mx.eval(x)
sp = bench(lambda: forward(x, B, layers, wout, with_attn=False))
sa = bench(lambda: forward(x, B, layers, wout, with_attn=True))
print(f"   无 attention: {sp*1e3:8.1f} ms   含: {sa*1e3:8.1f} ms   "
      f"attention 占 {(sa-sp)/sa*100:.0f}%  每窗 {(sa-sp)/B*1e3:.1f} ms")

print("\n-- 单 GEMM 在不同 batch 下的吞吐 --")
for B in (1, 4, 8):
    for (k, n, tag) in [(D, FF, "ff-up"), (FF, D, "ff-down")]:
        a = mx.random.normal((B * T, k)).astype(mx.float16)
        w = mx.random.normal((n, k)).astype(mx.float16)
        mx.eval(a, w)
        sec = bench(lambda: a @ w.T)
        print(f"   B={B} {tag:8s} ({B*T:5d},{k:4d})@({k:4d},{n:4d}): "
              f"{sec*1e3:7.3f} ms  {2*B*T*k*n/sec/1e12:5.2f} TFLOPS")
