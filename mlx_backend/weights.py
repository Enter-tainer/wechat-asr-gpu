"""模型目录 → MLX 权重张量（src/model.rs 的 load() 移植）。

.xnet 里的矩阵乘权重是 **int8 + per-tensor scale**（原版是 W8A8 的 int32 累加）。
Apple GPU 没有 int8 矩阵单元，所以这里在加载时一次性反量化成 fp16（或 fp32），
之后所有 GEMM 都用浮点；MLX 的 fp16 matmul 内部是 fp32 累加，实测相对误差 4.5e-4，
比 int8 激活量化本身的误差（~4e-3）还小一个数量级。
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from .xnet import Xnet

D = 512
HEADS = 8
HD = D // HEADS
FF = 2048
LAYERS = 40
F1, F2, F3 = 32, 64, 192
FLAT = F3 * 20          # 3840


def _deq(w_i8: np.ndarray, scale: float, dt) -> mx.array:
    """int8 → 浮点反量化（在 f32 里乘 scale 再降精度，比直接 f16 相乘准一点）。"""
    w = (w_i8.astype(np.float32) * np.float32(scale)).astype(np.float16)
    if dt == mx.float32:
        w = w.astype(np.float32)
    return mx.array(np.ascontiguousarray(w))


def _mat(xn: Xnet, name: str, n: int, k: int, dt) -> mx.array:
    """取 (n,k) 的 int8 权重并转置成 (k,n)，这样前向里可以直接 y @ W。"""
    w, s = xn.q8(name, n, k)
    return _deq(w, s, dt).T


def _f32(xn: Xnet, name: str, dims=None) -> np.ndarray:
    return xn.f32(name, dims)


def load(model_dir: str, dt=mx.float16) -> tuple[dict, list[str]]:
    d = Path(model_dir)
    xnet_path = next((p for p in sorted(d.iterdir()) if p.suffix == ".xnet"), None)
    if xnet_path is None:
        raise FileNotFoundError(f"{d} 里没有 .xnet 模型文件")
    xn = Xnet(str(xnet_path))

    vocab_file = d / "dict.decoder.utf8.txt"
    vocab = [ln.split(" ")[1] if " " in ln else "" for ln in
             vocab_file.read_text(encoding="utf-8").splitlines()]
    out_b = _f32(xn, "output_conv.bias")
    if len(vocab) != out_b.shape[0]:
        raise ValueError(f"词表 {len(vocab)} 与模型输出 {out_b.shape[0]} 不符")

    W: dict = {}
    # ---- 卷积前端（f32 权重，转成 MLX 的 (O, KH, KW, C) 布局）----
    c1 = _f32(xn, "frontend.conv2d.weight", (F1, 1, 5, 5))
    c2 = _f32(xn, "frontend.conv2d_1.weight", (F2, F1, 7, 7))
    c3 = _f32(xn, "frontend.conv2d_2.weight", (F3, F2, 3, 3))
    W["c1"] = mx.array(np.ascontiguousarray(c1.transpose(0, 2, 3, 1))).astype(dt)
    W["c2"] = mx.array(np.ascontiguousarray(c2.transpose(0, 2, 3, 1))).astype(dt)
    W["c3"] = mx.array(np.ascontiguousarray(c3.transpose(0, 2, 3, 1))).astype(dt)

    # ---- 前端 linear: 3840 → 512 ----
    W["lin"] = _mat(xn, "frontend.linear", D, FLAT, dt)
    W["lin_b"] = mx.array(_f32(xn, "frontend.linear.bias", (D,))).astype(dt)

    # ---- 40 层 ----
    layers = []
    for i in range(LAYERS):
        p = f"layers.{i}."
        # q/k/v 三个 (512,512) 预先拼成一个 (512,1536)，前向里只发一次 GEMM
        wqkv = mx.concatenate([
            _mat(xn, p + "attention.q_conv", D, D, dt),
            _mat(xn, p + "attention.k_conv", D, D, dt),
            _mat(xn, p + "attention.v_conv", D, D, dt),
        ], axis=1)
        layers.append({
            "pn_g": mx.array(_f32(xn, p + "prior_norm.weight", (D,))).astype(dt),
            "pn_b": mx.array(_f32(xn, p + "prior_norm.bias", (D,))).astype(dt),
            "mn_g": mx.array(_f32(xn, p + "middle_norm.weight", (D,))).astype(dt),
            "mn_b": mx.array(_f32(xn, p + "middle_norm.bias", (D,))).astype(dt),
            "wqkv": wqkv,
            "wo": _mat(xn, p + "attention.output_transform_conv", D, D, dt),
            "wf1": _mat(xn, p + "feed_forward.filter_conv", FF, D, dt),
            "f1b": mx.array(_f32(xn, p + "feed_forward.filter_conv.bias", (FF,))).astype(dt),
            "wf2": _mat(xn, p + "feed_forward.output_conv", D, FF, dt),
            "f2b": mx.array(_f32(xn, p + "feed_forward.output_conv.bias", (D,))).astype(dt),
        })
    W["layers"] = layers

    # ---- 输出层（词表，不做 %4 补齐；-1e30 的 padding 只对 cublasLt 必要）----
    W["out"] = _mat(xn, "output_conv", out_b.shape[0], D, dt)
    W["out_b"] = mx.array(out_b).astype(dt)
    mx.eval(
        [W["c1"], W["c2"], W["c3"], W["lin"], W["lin_b"], W["out"], W["out_b"]]
        + [v for l in layers for v in l.values()]
    )
    return W, vocab
