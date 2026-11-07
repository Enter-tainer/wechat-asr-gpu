"""声学模型前向：卷积前端 + 40 层 pre-norm Transformer + CTC 输出层（MLX 版）。

与 src/model.rs 的对应关系：
  Rust(CPU/SIMT)                    本文件
  ------------------------------------------------------------------
  frontend() conv1/2/3              self._frontend()  → mx.conv2d
  mm(quant(y3), lin)                self._linear()
  norm + quant + 6 个 int8 GEMM     直接 LN + fp16 matmul（int8 量化环节整个消失）
  attention()（0.125 缩放、8 头）    mx.fast.scaled_dot_product_attention
  gelu()（tanh 近似）               self._gelu()
  argmax                            mx.argmax

数值口径（必须一致）：LN eps=1e-6、attention scale=0.125=1/sqrt(64)、
tanh-GELU 的 0.7978846/0.044715、conv 的时间/频率 padding 不对称（见 _frontend）。
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from .weights import D, FF, FLAT, HD, HEADS, LAYERS, load


def _relu(x):
    """mlx.core 没有 mx.relu（那是 mlx.nn 的），用 maximum 等价替换。"""
    return mx.maximum(x, 0)


class Model:
    def __init__(self, model_dir: str, dt=mx.float16):
        self.dt = dt
        self.W, self.vocab = load(model_dir, dt)
        self.layers = self.W["layers"]

    # ---------------- 前端 ----------------
    def _frontend(self, feats: np.ndarray) -> tuple[int, mx.array]:
        """[t][40] 归一化特征 → [t1][3840]。

        conv1: 5×5, 1→32，时间/频率各补 2，输出 [t][40][32]
        conv2: 7×7, 32→64，时间 stride 5 补 1；频率补 3（左）4（右）→ 输出 41 列
        conv3: 3×3, 64→192，频率 stride 2，时间补 1 → 输出 [t1][20][192]
        """
        t = feats.shape[0]
        t1 = (t - 5) // 5 + 1
        x = mx.array(feats[None, :, :, None].astype(np.float16))
        x = _relu(mx.conv2d(mx.pad(x, [(0, 0), (2, 2), (2, 2), (0, 0)]), self.W["c1"]))
        x = _relu(mx.conv2d(mx.pad(x, [(0, 0), (1, 1), (3, 4), (0, 0)]),
                            self.W["c2"], stride=(5, 1)))          # (1,t1,41,64)
        x = _relu(mx.conv2d(mx.pad(x, [(0, 0), (1, 1), (0, 0), (0, 0)]),
                            self.W["c3"], stride=(1, 2)))          # (1,t1,20,192)
        # (t1,20,192) → (t1,192,20) → (t1, 3840)，对应 Rust 的 [o*20 + f] 展开顺序
        return t1, x.reshape(t1, 20, 192).transpose(0, 2, 1).reshape(t1, FLAT)

    # ---------------- 小算子 ----------------
    def _frontend_batch(self, X: np.ndarray, t1: int) -> mx.array:
        """批量卷积前端：X (B,t,40) → (B*t1, 3840)。"""
        return self._frontend_batch_mx(mx.array(X[..., None].astype(np.float16)), t1)

    def _frontend_batch_mx(self, x: mx.array, t1: int) -> mx.array:
        """同上，但接收已经是 mx.array 的 (B,t,40,1)（便于 mx.compile 捕获）。"""
        B = x.shape[0]
        x = _relu(mx.conv2d(mx.pad(x, [(0, 0), (2, 2), (2, 2), (0, 0)]), self.W["c1"]))
        x = _relu(mx.conv2d(mx.pad(x, [(0, 0), (1, 1), (3, 4), (0, 0)]),
                            self.W["c2"], stride=(5, 1)))          # (B,t1,41,64)
        x = _relu(mx.conv2d(mx.pad(x, [(0, 0), (1, 1), (0, 0), (0, 0)]),
                            self.W["c3"], stride=(1, 2)))          # (B,t1,20,192)
        return x.transpose(0, 1, 3, 2).reshape(B * t1, FLAT)

    def _ln(self, x: mx.array, g: mx.array, b: mx.array) -> mx.array:
        """LayerNorm 在 fp32 里算（与 Rust 一致，eps=1e-6），再回 fp16。"""
        xf = x.astype(mx.float32)
        mu = mx.mean(xf, axis=-1, keepdims=True)
        var = mx.mean((xf - mu) ** 2, axis=-1, keepdims=True)
        y = (xf - mu) * mx.rsqrt(var + 1e-6) * g.astype(mx.float32) + b.astype(mx.float32)
        return y.astype(x.dtype)

    @staticmethod
    def _gelu(x: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        return (0.5 * xf * (1.0 + mx.tanh(0.7978846 * (xf + 0.044715 * xf * xf * xf)))).astype(x.dtype)

    def _attn(self, y: mx.array, t1: int, p: dict) -> mx.array:
        qkv = (y @ p["wqkv"]).reshape(t1, 3, HEADS, HD)
        q, k, v = mx.split(qkv, 3, axis=1)                       # (t1,1,H,HD)
        q = q.reshape(t1, HEADS, HD).transpose(1, 0, 2)[None]
        k = k.reshape(t1, HEADS, HD).transpose(1, 0, 2)[None]
        v = v.reshape(t1, HEADS, HD).transpose(1, 0, 2)[None]
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
        return a[0].transpose(1, 0, 2).reshape(t1, D)

    # ---------------- 整窗前向 ----------------
    def forward(self, feats: np.ndarray) -> list[int]:
        """feats: (t,40) float32 → 每个 50ms 帧的 argmax token id。"""
        t = feats.shape[0]
        if t < 5:
            return []
        t1, y3 = self._frontend(np.asarray(feats, dtype=np.float32))
        h = y3 @ self.W["lin"] + self.W["lin_b"]
        for p in self.layers:
            y = self._ln(h, p["pn_g"], p["pn_b"])
            h = h + self._attn(y, t1, p) @ p["wo"]
            y = self._ln(h, p["mn_g"], p["mn_b"])
            f = self._gelu(y @ p["wf1"] + p["f1b"])
            h = h + f @ p["wf2"] + p["f2b"]
        lo = h @ self.W["out"] + self.W["out_b"]
        ids = mx.argmax(lo, axis=-1)
        mx.eval(ids)
        return [int(i) for i in np.asarray(ids)]

    # ---------------- CTC 折叠 ----------------
    def text(self, ids: list[int]) -> str:
        """去重 + 去 blank；<SPACE>/<UNK>/|词首 的特判与 Rust 一致。"""
        out: list[str] = []
        prev = -1
        for i in ids:
            if i != prev and i != 0:
                tok = self.vocab[i] if 0 <= i < len(self.vocab) else ""
                if tok == "<UNK>":
                    pass
                elif tok == "<SPACE>":
                    out.append(" ")
                elif len(tok) > 1 and tok.startswith("|"):
                    if out and out[-1][-1:].isascii() and out[-1][-1:].isalnum():
                        out.append(" ")
                    out.append(tok[1:])
                else:
                    out.append(tok)
            prev = i
        return "".join(out)
