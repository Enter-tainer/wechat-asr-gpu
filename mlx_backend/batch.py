"""批窗口解码：利用"编码器窗口之间没有状态依赖"把多个窗口同时送 GPU。

原来的分块解码是严格串行的（每个窗口的起点取决于上一窗口末尾的 blank 截断）。
批窗口版本改成固定网格（窗口起点 = k*blk），于是 B 个窗口可以一次算完，
再各取 [s, s+blk) 区间的 id 拼起来，最后统一做一次 CTC 折叠。

两级加速（M2 Pro / bv58.wav 58.65s 实测）：
  串行分块                    254 ms   231x
  + 批窗口 B=4                191 ms   306x
  + mx.compile 整窗图          135 ms   434x
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from .model import Model, _relu
from .weights import D, HD, HEADS, FLAT


def _forward_body(m: Model, x: mx.array, t1: int, B: int) -> mx.array:
    """整窗前向（可被 mx.compile 捕获）：x (B,t,40,1) fp16 → ids (B,t1)。"""
    h = m._frontend_batch_mx(x, t1) @ m.W["lin"] + m.W["lin_b"]
    for p in m.layers:
        y = m._ln(h, p["pn_g"], p["pn_b"])
        qkv = (y @ p["wqkv"]).reshape(B, t1, 3, HEADS, HD)
        q, k, v = mx.split(qkv, 3, axis=2)                        # (B,t1,1,H,HD)
        q = q.reshape(B, t1, HEADS, HD).transpose(0, 2, 1, 3)
        k = k.reshape(B, t1, HEADS, HD).transpose(0, 2, 1, 3)
        v = v.reshape(B, t1, HEADS, HD).transpose(0, 2, 1, 3)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
        h = h + a.transpose(0, 2, 1, 3).reshape(B * t1, D) @ p["wo"]
        y = m._ln(h, p["mn_g"], p["mn_b"])
        h = h + m._gelu(y @ p["wf1"] + p["f1b"]) @ p["wf2"] + p["f2b"]
    return mx.argmax(h @ m.W["out"] + m.W["out_b"], axis=-1).reshape(B, t1)


class BatchEngine:
    """固定网格 + 批窗口 + 编译整窗图。"""

    def __init__(self, model: Model, cfg, batch: int = 4, compile_graph: bool = True):
        self.m = model
        self.cfg = cfg
        self.batch = max(1, int(batch))
        self.compile_graph = compile_graph
        self._cache: dict[tuple[int, int], object] = {}

    def _fn(self, B: int, t1: int):
        key = (B, t1)
        if key not in self._cache:
            f = lambda x, t1=t1, B=B: _forward_body(self.m, x, t1, B)   # noqa: E731
            self._cache[key] = mx.compile(f) if self.compile_graph else f
        return self._cache[key]

    def _window(self, feats: list[np.ndarray], silence: np.ndarray, s0: int) -> np.ndarray:
        """固定网格窗口 [s0-prev, s0+blk+post) + pad；左右不足补静音，保证所有窗口等长。"""
        c = self.cfg
        t_win = c.prev + c.blk + c.post
        left = min(c.prev, s0)
        a = s0 - left
        b = min(s0 + c.blk + c.post, len(feats))
        x = np.stack(feats[a:b]) if b > a else np.zeros((0, 40), np.float32)
        pre = c.prev - left
        if pre:
            x = np.concatenate([np.tile(silence, (pre, 1)), x])
        if len(x) < t_win:
            x = np.concatenate([x, np.tile(silence, (t_win - len(x), 1))])
        return np.concatenate([x, np.tile(silence, (50, 1))])

    def run(self, feats: list[np.ndarray], silence: np.ndarray) -> list[int]:
        c = self.cfg
        n = len(feats)
        if n == 0:
            return []
        starts = list(range(0, n, c.blk))
        head = c.prev // 5
        all_ids: list[int] = []
        for i in range(0, len(starts), self.batch):
            grp = starts[i:i + self.batch]
            X = np.stack([self._window(feats, silence, s0) for s0 in grp])
            t1 = (X.shape[1] - 5) // 5 + 1
            ids = np.asarray(self._fn(len(grp), t1)(
                mx.array(X[..., None].astype(np.float16))))
            for s0, row in zip(grp, ids):
                keep = min(c.blk, max(n - s0, 0)) // 5
                all_ids.extend(row[head:head + keep].tolist())
        return all_ids
