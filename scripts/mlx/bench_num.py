"""量化误差实验：同一次 int8 量化，三条路径的数值差异。

路径 A（现有 CPU/CUDA 口径，视为"参考"）：int8×int8 的 int32 累加，再 (acc*xs)*ws+bias
路径 B（MLX fp16 移植路线）：权重/激活反量化成 fp16 后 mx.matmul
路径 C（fp32 反量化）：同上但 fp32，用来看"只换精度不换累加"的误差

注意：k=512 时 fp32 里的整数累加是精确的（|acc|≤512*127²=8.3M < 2^24），
所以 A 用 fp32 matmul 就能等价模拟 int32 累加。
"""
import mlx.core as mx

D, FF, V = 512, 2048, 10504
T = 1600


def rel_err(y, ref):
    y = y.astype(mx.float32)
    ref = ref.astype(mx.float32)
    return float(mx.max(mx.abs(y - ref))), float(mx.sqrt(mx.mean((y - ref) ** 2)) / mx.sqrt(mx.mean(ref ** 2)))


mx.random.seed(0)


def check(t, k, n, tag):
    # 真实风格数据：权重 int8 per-tensor，激活 per-row 动态 int8
    w_i8 = mx.random.randint(-127, 128, (n, k)).astype(mx.int8)
    ws = 0.0009
    x_f = mx.random.normal((t, k)).astype(mx.float32) * 0.5
    m = mx.max(mx.abs(x_f), axis=1, keepdims=True)
    xs = m / 127.0
    xq = mx.round(x_f / xs).astype(mx.int8)

    # A: 精确 int32 累加（fp32 承载整数，k<=512 时精确）
    acc = xq.astype(mx.float32) @ w_i8.astype(mx.float32).T
    y_a = acc * xs * ws

    # B: fp16 反量化 + fp16 matmul（MLX 移植最省事的写法）
    x16 = (xq.astype(mx.float16) * xs.astype(mx.float16))
    w16 = (w_i8.astype(mx.float16) * ws)
    y_b = x16 @ w16.T

    # B2: fp32 累加（fp16 存、fp32 算）
    y_b2 = xq.astype(mx.float16).astype(mx.float32) @ w_i8.astype(mx.float16).astype(mx.float32).T * (xs * ws)

    # C: fp32 反量化 + fp32 matmul
    y_c = (xq.astype(mx.float32) * xs) @ (w_i8.astype(mx.float32) * ws).T

    mx.eval(y_a, y_b, y_b2, y_c)
    ma, ra = rel_err(y_b, y_a)
    ma2, ra2 = rel_err(y_b2, y_a)
    mc, rc = rel_err(y_c, y_a)
    print(f"{tag:22s} t={t} k={k:4d} n={n:5d}")
    print(f"   fp16 反量化+matmul : max_abs={ma:.3e}  rel_rms={ra:.3e}")
    print(f"   fp16 存/fp32 累加  : max_abs={ma2:.3e}  rel_rms={ra2:.3e}")
    print(f"   fp32 反量化+matmul : max_abs={mc:.3e}  rel_rms={rc:.3e}")


check(T, 512, 2048, "FF up")
check(T, 2048, 512, "FF down")
check(T, 512, 512, "q/o proj")
check(T, 512, V, "output layer")
check(T, 3840, 512, "linear k=3840")

print("\n参考量级：现有 CUDA vs CPU 的 attention 对拍误差 1.2e-7（fp32 求和顺序），"
      "int8 激活量化本身的相对误差 ~1/254=4e-3。")
print("另测：fp16 输入 + fp16 输出时，MLX matmul 内部是否 fp32 累加 -> 看上面第 1 行 rel_rms")
print("      ~1e-3 量级 => 输出舍入为主（内部 fp32 累加）；~1e-2 量级 => 内部也在 fp16 里累加。")
