"""官方 .xnet 模型文件解析（src/xnet.rs 的 Python 移植）。

.xnet 是一个自定义容器：文件头部是 "XNET" + 若干元数据，**尾部**是一张权重/常量记录表，
按名称索引，每条记录带 dtype、quant scale、shape 和紧跟其后的裸数据；记录表一直链到
"零长度名 + 全零填充" 为止。记录表起点的定位办法与 Rust 一致：找一个
"u32 长度 n（1..128）+ 3 个 0 字节 + n 个可打印 ASCII" 的位置，然后试着把整张表解析完，
能恰好解析到文件尾的就是正确的起点。
"""
from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass

import numpy as np

F32 = 11
I8 = 6

_ELEM = {I8: 1, 2: 1, 3: 1, 4: 2, 5: 2, 1: 4, 7: 4, 8: 4, F32: 4, 9: 8, 10: 8, 12: 8}


@dataclass
class Rec:
    dtype: int
    scale: float | None
    dims: tuple[int, ...]
    off: int
    len: int


class _Eof(Exception):
    pass


def _chain(d, p: int) -> dict[str, Rec] | None:
    """从 p 起逐条解析，必须恰好解析到文件尾（零长度名 + 全零填充）才算成功。"""
    ln_total = len(d)
    out: dict[str, Rec] = {}

    def u32() -> int:
        nonlocal p
        if p + 4 > ln_total:
            raise _Eof
        v = struct.unpack_from("<I", d, p)[0]
        p += 4
        return v

    def take(n: int) -> bytes:
        nonlocal p
        if n < 0 or p + n > ln_total:
            raise _Eof
        s = d[p:p + n]
        p += n
        return s

    try:
        while True:
            n = u32()
            if n == 0:
                rest = d[p:]
                if len(rest) <= 16 and not any(rest) and len(out) > 100:
                    return out
                return None
            if n > 256:
                return None
            raw = take(n)
            try:
                name = raw.decode("utf-8")
            except UnicodeDecodeError:
                return None
            dtype = u32()
            if dtype not in _ELEM:
                return None
            es = _ELEM[dtype]
            quant = u32()
            u32()
            u32()
            u32()
            ns = u32()
            u32()
            if ns > 4096:
                return None
            sc = take(ns * 4)
            scale = struct.unpack_from("<f", sc, 0)[0] if (quant != 0 and ns > 0) else None
            nz = u32()
            take(nz)
            nd = u32()
            if nd > 8:
                return None
            dims = tuple(u32() for _ in range(nd))
            ln = u32()
            if ln != int(np.prod(dims)) * es:
                return None
            out[name] = Rec(dtype=dtype, scale=scale, dims=dims, off=p, len=ln)
            take(ln)
    except _Eof:
        return None


def parse(d) -> dict[str, Rec]:
    if bytes(d[:4]) != b"XNET":
        raise ValueError("不是 xnet 模型文件")
    b = np.frombuffer(d, dtype=np.uint8)
    n = len(b)
    if n < 16:
        raise ValueError("xnet 文件过短")
    # 候选起点：u32 长度在 1..128 且后面 3 个字节都是 0
    z = (b[1:n - 3] == 0) & (b[2:n - 2] == 0) & (b[3:n - 1] == 0)
    ln0 = b[:n - 4]
    cand = np.nonzero(z & (ln0 >= 1) & (ln0 <= 128))[0]
    for at in cand.tolist():
        if at < 8:
            continue
        nl = int(b[at])
        if at + 4 + nl > n:
            continue
        name = bytes(b[at + 4:at + 4 + nl])
        if not all(0x21 <= c <= 0x7E for c in name):
            continue
        m = _chain(d, at)
        if m is not None:
            return m
    raise ValueError("找不到 xnet 权重表")


class Xnet:
    """mmap 打开 .xnet，按名字取 numpy 视图。"""

    def __init__(self, path: str):
        self._f = open(path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.recs = parse(self._mm)
        self._buf = np.frombuffer(self._mm, dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.recs)

    def has(self, name: str) -> bool:
        return name in self.recs

    def raw(self, name: str) -> np.ndarray:
        r = self.recs[name]
        return self._buf[r.off:r.off + r.len]

    def f32(self, name: str, dims: tuple[int, ...] | None = None) -> np.ndarray:
        r = self.recs[name]
        if r.dtype != F32:
            raise ValueError(f"{name}: dtype={r.dtype} 不是 f32")
        if dims is not None and tuple(r.dims) != tuple(dims):
            raise ValueError(f"{name}: 形状 {r.dims} != {dims}")
        return np.frombuffer(self._buf[r.off:r.off + r.len].tobytes(), dtype="<f4").reshape(r.dims)

    def q8(self, name: str, n: int, k: int) -> tuple[np.ndarray, float]:
        """int8 权重 + per-tensor 量化 scale。"""
        full = f"{name}.weight_quantized_s8"
        r = self.recs[full]
        if r.dtype != I8 or (len(r.dims) >= 2 and tuple(r.dims[:2]) != (n, k)) or r.len != n * k:
            raise ValueError(f"{full}: 形状 {r.dims} != ({n},{k})")
        if r.scale is None:
            raise ValueError(f"{full}: 缺少量化 scale")
        w = np.frombuffer(self._buf[r.off:r.off + r.len].tobytes(), dtype=np.int8)
        return w.reshape(n, k), float(r.scale)

    def close(self) -> None:
        self._buf = None
        self._mm.close()
        self._f.close()
