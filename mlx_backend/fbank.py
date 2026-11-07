"""HTK 风格 FBank 特征（src/fbank.rs 的 Python 移植，逐位对齐口径）。

39 维 log-mel + 对数能量 = 40 维；25ms 窗 / 10ms 帧移；减去最近 0.5s 的滑动均值（只减均值、不除方差）。
注意：这里的"能量"直接用 i16 原始幅值算，不做 /32768 归一化 —— 幅度常量会被滑动均值部分吃掉，
但为了和 Rust/CUDA 版逐位对拍，必须保持同样的口径。
"""
from __future__ import annotations

import numpy as np

WIN = 400
HOP = 160
NFFT = 512
NCH = 39
DIM = NCH + 1
CMS_WIN = 50
SR = 16000

_PREEMPH = 0.97
_FLOOR = 1.0


def _tables() -> tuple[np.ndarray, np.ndarray]:
    pi = np.pi
    n = np.arange(WIN, dtype=np.float64)
    ham = 0.54 - 0.46 * np.cos(2.0 * pi * n / (WIN - 1))

    def mel(f):
        return 1127.0 * np.log1p(np.asarray(f, dtype=np.float64) / 700.0)

    hi = float(mel(8000.0))
    cf = hi * np.arange(1, NCH + 2, dtype=np.float64) / (NCH + 1)   # 40 个三角中心（mel 域）
    M = np.zeros((NFFT // 2, DIM), dtype=np.float64)                # (256 bins, 40)
    for k in range(1, NFFT // 2 + 1):
        m = float(mel(k * SR / NFFT))
        c = int(np.searchsorted(cf, m, side="left"))                # cf 中 < m 的个数
        if c == 0:
            M[k - 1, 0] += m / cf[0]
        elif c < NCH:
            w = (cf[c] - m) / (cf[c] - cf[c - 1])
            M[k - 1, c - 1] += w
            M[k - 1, c] += 1.0 - w
        elif c == NCH:
            M[k - 1, NCH - 1] += (cf[NCH] - m) / (cf[NCH] - cf[NCH - 1])
        # c > NCH: 超出 8kHz 的 bin，全 0
    return ham, M


class Fbank:
    def __init__(self) -> None:
        self._ham, self._mel = _tables()
        self.buf = np.zeros(0, dtype=np.int16)
        self.hist: list[np.ndarray] = []
        self.sum = np.zeros(DIM, dtype=np.float64)

    # ---- 单帧 / 多帧 ----
    def _frames(self, nfr: int) -> np.ndarray:
        idx = np.arange(nfr)[:, None] * HOP + np.arange(WIN)[None, :]
        x = self.buf[idx].astype(np.float64)                 # (nfr, WIN)
        e = (x * x).sum(axis=1)                              # 对数能量的分子
        pe = np.empty_like(x)
        pe[:, 0] = x[:, 0] * (1.0 - _PREEMPH)
        pe[:, 1:] = x[:, 1:] - _PREEMPH * x[:, :-1]
        spec = np.fft.rfft(pe * self._ham, n=NFFT)            # (nfr, 257)
        p = spec[:, 1:].real ** 2 + spec[:, 1:].imag ** 2     # (nfr, 256) 对应 bin 1..256
        out = p @ self._mel                                  # (nfr, 40)
        out[:, :NCH] = np.log(np.maximum(out[:, :NCH], _FLOOR))
        out[:, NCH] = np.log(np.maximum(e, 1e-10))
        return out

    def _sub(self, f: np.ndarray) -> np.ndarray:
        n = max(len(self.hist), 1)
        return (f - self.sum / n).astype(np.float32)

    def push(self, pcm: np.ndarray, out: list[np.ndarray]) -> None:
        self.buf = np.concatenate([self.buf, np.asarray(pcm, dtype=np.int16)])
        nfr = (len(self.buf) - WIN) // HOP + 1 if len(self.buf) >= WIN else 0
        if nfr <= 0:
            return
        raws = self._frames(nfr)
        for f in raws:
            if len(self.hist) == CMS_WIN:
                self.sum -= self.hist.pop(0)
            self.hist.append(f)
            self.sum += f
            out.append(self._sub(f))
        self.buf = self.buf[nfr * HOP:]

    def silence(self) -> np.ndarray:
        f = np.zeros(DIM, dtype=np.float64)
        f[NCH] = np.log(1e-10)
        return self._sub(f)
