"""流式/批模式分块解码（src/stream.rs 的 Python 移植）。

批模式不需要中间结果：块长直接决定吞吐。每块的上下文开销 = prev + post + pad，
有效计算量 ≈ (prev + blk + post + pad) / blk。默认 1600 帧时重算放大 ≈1.22x。
"""
from __future__ import annotations

import os

import numpy as np

from .fbank import DIM, Fbank

PREV_DEFAULT = 200
BLK_DEFAULT = 400
POST_DEFAULT = 100
PAD = 50                 # 窗口末尾补的静音帧
WHOLE_MAX = 2000         # 整段解码上限（特征帧）：20 秒

PUNCT = set("，。？！、,.?!")


def _env(k: str, d: int) -> int:
    for name in (k, k.replace("MLX_", "WTL_", 1)):
        v = os.environ.get(name)
        if v:
            try:
                n = int(v)
                if n >= 100:
                    return n
            except ValueError:
                pass
    return d


class Cfg:
    def __init__(self):
        self.prev = _env("MLX_PREV", PREV_DEFAULT)
        self.blk = _env("MLX_BLK", BLK_DEFAULT)
        self.post = _env("MLX_POST", POST_DEFAULT)


class Stream:
    def __init__(self, model, cfg: Cfg | None = None):
        self.m = model
        self.cfg = cfg or Cfg()
        self.fb = Fbank()
        self.feats: list[np.ndarray] = []
        self.ids: list[int] = []
        self.s = 0

    # ---- 特征 ----
    def push(self, pcm: np.ndarray) -> None:
        self.fb.push(pcm, self.feats)
        while len(self.feats) >= self.s + self.cfg.blk + self.cfg.post:
            self.commit()

    def _feats_slice(self, a: int, end: int, pad: int) -> np.ndarray:
        x = np.stack(self.feats[a:end]) if end > a else np.zeros((0, DIM), dtype=np.float32)
        if pad:
            x = np.concatenate([x, np.tile(self.fb.silence(), (pad, 1))], axis=0)
        return x

    def decode(self, prev: int, end: int, pad: int) -> list[int]:
        a = max(self.s - prev, 0)
        out = self.m.forward(self._feats_slice(a, end, pad))
        return out[(self.s - a) // 5:]

    def commit(self) -> None:
        ids = self.decode(self.cfg.prev, self.s + self.cfg.blk + self.cfg.post, PAD)
        ids = ids[: self.cfg.blk // 5]
        # 在块尾附近的连续 blank 处截断，避免字恰好落在块边界被两边都丢掉
        for c in range(len(ids) - 1, max(len(ids) // 2, 1) - 1, -1):
            if ids[c] == 0 and ids[c - 1] == 0:
                ids = ids[:c]
                break
        self.s += len(ids) * 5
        self.ids.extend(ids)

    # ---- 出结果 ----
    def finish(self) -> str:
        if len(self.feats) <= WHOLE_MAX:
            return tidy(self.m.text(self.m.forward(np.stack(self.feats))))
        if len(self.feats) > self.s:
            self.ids.extend(self.decode(self.cfg.prev, len(self.feats), PAD))
        return tidy(self.m.text(self.ids))

    def partial(self) -> str:
        ids = list(self.ids)
        if len(self.feats) > self.s:
            ids.extend(self.decode(self.cfg.prev // 2, len(self.feats), PAD))
        return tidy(self.m.text(ids)).rstrip("".join(PUNCT))


def tidy(t: str) -> str:
    """去掉开头的标点、以及「，。」这类连续标点里靠前的那个。"""
    c = list(t.lstrip("".join(PUNCT)))
    out = [ch for i, ch in enumerate(c)
           if not (ch in PUNCT and i + 1 < len(c) and c[i + 1] in PUNCT)]
    return "".join(out)
