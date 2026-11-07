"""在 M2 Pro 上用 MLX 复刻 wtmodel 的算子形状，实测 fp16 前向速度 → RTF。

形状来源（wechat-asr-gpu/README.md + TECH-42）：
  40 层 pre-norm Transformer, D=512, FF=2048, 8 头 (head_dim=64), 词表 10504
  blk=1600 帧 = 16s 音频/窗口；流式窗口 t=400
"""
import time
import mlx.core as mx

D, FF, L, H, V = 512, 2048, 40, 8, 10504
DH = D // H
T_BATCH = 1600   # 批模式 16s/窗
T_STREAM = 400   # 流式窗口 4s/窗
REPS = 5


def bench(fn, reps=REPS, warm=2):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / reps


def matmul_tflops(t, k, n, dt=mx.float16):
    x = mx.random.normal((t, k)).astype(dt)
    w = mx.random.normal((n, k)).astype(dt)
    mx.eval(x, w)
    sec = bench(lambda: x @ w.T)
    flops = 2.0 * t * k * n
    return sec, flops / sec / 1e12


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


def layernorm(x, g, b, eps=1e-5):
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mu) ** 2, axis=-1, keepdims=True)
    return (x - mu) * mx.rsqrt(var + eps) * g + b


def forward(x, T, layers, wout):
    h = x
    for p in layers:
        y = layernorm(h, p["g1"], p["b1"])
        qkv = (y @ p["wqkv"].T).reshape(T, 3, H, DH)
        q, k, v = mx.split(qkv, 3, axis=1)          # (T,1,H,DH)
        q = q.reshape(T, H, DH).transpose(1, 0, 2)[None]   # (1,H,T,DH)
        k = k.reshape(T, H, DH).transpose(1, 0, 2)[None]
        v = v.reshape(T, H, DH).transpose(1, 0, 2)[None]
        att = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / DH ** 0.5)
        att = att[0].transpose(1, 0, 2).reshape(T, D)
        h = h + att @ p["wo"].T
        y = layernorm(h, p["g2"], p["b2"])
        f = mx.maximum(y @ p["w1"].T, 0)          # 近似 GELU
        h = h + f @ p["w2"].T
    return h @ wout.T


mx.random.seed(0)
print(f"MLX {mx.__version__}  device={mx.default_device()}")
print("== 单 GEMM fp16 吞吐 ==")
for (t, k, n, tag) in [(T_BATCH, D, 3 * D, "qkv-1600"), (T_BATCH, D, FF, "ff-up-1600"),
                       (T_BATCH, FF, D, "ff-down-1600"), (T_BATCH, D, D, "o-1600"),
                       (T_STREAM, D, FF, "ff-up-400")]:
    sec, tfl = matmul_tflops(t, k, n)
    print(f"  {tag:14s} t={t:4d} k={k:4d} n={n:5d}: {sec*1e3:7.3f} ms  {tfl:5.2f} TFLOPS")

layers, wout = make_model()
for T in (T_BATCH, T_STREAM):
    x = (mx.random.normal((T, D)) * 0.1).astype(mx.float16)
    mx.eval(x)

    def run(T=T, x=x):
        return forward(x, T, layers, wout)

    sec = bench(run, reps=3, warm=1)
    audio_s = T / 100.0
    flops = 2.0 * T * L * (4 * D * D + 2 * D * FF) + 2.0 * T * V * D
    print(f"== 整窗 T={T}（{audio_s:.0f}s 音频）: {sec*1e3:8.1f} ms  "
          f"RTF={sec/audio_s:.4f}  ({audio_s/sec:.1f}x 实时)  "
          f"等效 {flops/sec/1e12:.2f} TFLOPS")
    if T == T_BATCH:
        compiled = mx.compile(run)
        sec_c = bench(compiled, reps=3, warm=2)
        print(f"   mx.compile 后: {sec_c*1e3:8.1f} ms  RTF={sec_c/audio_s:.4f}  ({audio_s/sec_c:.1f}x 实时)")

print("== 权重 int8 + fp16 激活（quantized_matmul, group=64, bits=8）==")
try:
    for (t, k, n) in [(T_BATCH, D, FF), (T_BATCH, D, 3 * D)]:
        x = mx.random.normal((t, k)).astype(mx.float16)
        w = (mx.random.normal((n, k)) * 0.02).astype(mx.float16)
        wq, scales, biases = mx.quantize(w, group_size=64, bits=8)
        mx.eval(wq, scales, biases)
        sec_q = bench(lambda: mx.quantized_matmul(x, wq, scales, biases, transpose=True,
                                                  group_size=64, bits=8))
        sec_d = bench(lambda: x @ w.T)
        print(f"  t={t} k={k} n={n}: int8w {sec_q*1e3:7.3f} ms vs fp16w {sec_d*1e3:7.3f} ms "
              f"({sec_d/sec_q:.2f}x)")
except Exception as e:
    print("  quantized_matmul 不可用:", type(e).__name__, e)
