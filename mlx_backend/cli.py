"""wasr-mlx —— 与 Rust 版 `wasr` 对齐的批模式 CLI（Apple Silicon / MLX 后端）。

用法：
  python -m mlx_backend.cli <模型目录> [--blk N] [--batch N] [--fp32] [--json FILE] [--txt DIR] a.wav [b.wav ...]

stdout 为 Tab 分隔：`名字  音频秒  解码毫秒  RTF  文本`；stderr 打加载信息与批聚合吞吐。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: 只支持 16-bit PCM")
        if w.getnchannels() != 1:
            raise ValueError(f"{path}: 只支持单声道（当前 {w.getnchannels()} 声道）")
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm, sr


def resample_to_16k(pcm: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return pcm
    n = int(round(len(pcm) * 16000 / sr))
    x = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm.astype(np.float32))
    return np.clip(x, -32768, 32767).astype(np.int16)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wasr-mlx", description="微信离线 ASR（MLX / Apple Silicon）")
    ap.add_argument("model_dir")
    ap.add_argument("--blk", type=int, default=1600, help="解码块长（特征帧，10ms/帧），默认 1600")
    ap.add_argument("--batch", type=int, default=4, help="一次送 GPU 的窗口数，默认 4")
    ap.add_argument("--prev", type=int, default=None)
    ap.add_argument("--post", type=int, default=None)
    ap.add_argument("--fp32", action="store_true", help="权重/计算用 fp32（更准，约慢 1.6x）")
    ap.add_argument("--no-compile", action="store_true", help="不用 mx.compile（调试用）")
    ap.add_argument("--json", default=None)
    ap.add_argument("--txt", default=None)
    ap.add_argument("--selftest", action="store_true", help="只跑形状自测（随机权重）后退出")
    ap.add_argument("files", nargs="*")
    a = ap.parse_intermixed_args(argv)

    try:
        import mlx.core as mx
    except ImportError:
        print("需要 MLX：python3 -m pip install --user mlx numpy", file=sys.stderr)
        return 1

    if a.selftest:
        return _selftest()

    import os
    os.environ["MLX_BLK"] = str(a.blk)
    if a.prev:
        os.environ["MLX_PREV"] = str(a.prev)
    if a.post:
        os.environ["MLX_POST"] = str(a.post)

    from .batch import BatchEngine
    from .decode import Cfg, tidy
    from .fbank import Fbank
    from .model import Model

    t0 = time.perf_counter()
    m = Model(a.model_dir, dt=mx.float32 if a.fp32 else mx.float16)
    load_ms = (time.perf_counter() - t0) * 1e3
    cfg = Cfg()
    eng = BatchEngine(m, cfg, a.batch, compile_graph=not a.no_compile)
    print(f"load {load_ms:.0f}ms  backend=MLX({mx.default_device()})  "
          f"dtype={'fp32' if a.fp32 else 'fp16'}  blk={cfg.blk}  batch={a.batch}  "
          f"vocab={len(m.vocab)}", file=sys.stderr)

    if not a.files:
        print("没有输入音频", file=sys.stderr)
        return 2

    rows = []
    wall0 = time.perf_counter()
    tot_audio = tot_ms = 0.0
    for f in a.files:
        name = Path(f).stem
        try:
            pcm, sr = read_wav(f)
            pcm = resample_to_16k(pcm, sr)
        except Exception as e:                      # noqa: BLE001
            print(f"跳过 {f}: {e}", file=sys.stderr)
            continue
        audio = len(pcm) / 16000.0
        t = time.perf_counter()
        fb = Fbank()
        feats: list[np.ndarray] = []
        fb.push(pcm, feats)
        text = tidy(m.text(eng.run(feats, fb.silence())))
        ms = (time.perf_counter() - t) * 1e3
        tot_audio += audio
        tot_ms += ms
        print(f"{name}\t{audio:.2f}\t{ms:.0f}\t{ms / 1000.0 / audio:.4f}\t{text}")
        if a.txt:
            Path(a.txt).mkdir(parents=True, exist_ok=True)
            (Path(a.txt) / f"{name}.txt").write_text(text, encoding="utf-8")
        rows.append({"name": name, "audio_s": round(audio, 2), "decode_ms": round(ms),
                     "rtf": round(ms / 1000.0 / audio, 4), "text": text})

    wall = time.perf_counter() - wall0
    print(f"\n批模式: {len(rows)} 条, 音频 {tot_audio:.1f}s, 解码合计 {tot_ms / 1e3:.2f}s, "
          f"批墙钟 {wall:.2f}s, 聚合 {tot_audio / wall:.1f}x 实时, "
          f"平均 RTF {tot_ms / 1e3 / max(tot_audio, 1e-9):.4f}", file=sys.stderr)
    if a.json:
        Path(a.json).write_text(json.dumps({
            "backend": "mlx", "dtype": "fp32" if a.fp32 else "fp16", "blk": cfg.blk,
            "batch": a.batch, "load_ms": round(load_ms),
            "total_audio_s": round(tot_audio, 2), "total_decode_s": round(tot_ms / 1e3, 2),
            "batch_wall_s": round(wall, 2), "agg_x_realtime": round(tot_audio / wall, 1),
            "files": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 已写入 {a.json}", file=sys.stderr)
    return 0


def _selftest() -> int:
    """不开模型也能验形状：随机权重跑通 conv 前端 → 40 层 → argmax。"""
    import mlx.core as mx
    from .fbank import DIM
    from .model import Model as _M
    from .weights import D, FF, FLAT, HD, HEADS

    m = object.__new__(_M)
    m.dt = mx.float16
    rng = np.random.default_rng(0)
    def s(*shape):
        return mx.array((rng.standard_normal(shape) * 0.02).astype(np.float16))
    m.layers = [{
        "pn_g": mx.ones((D,), mx.float16), "pn_b": mx.zeros((D,), mx.float16),
        "mn_g": mx.ones((D,), mx.float16), "mn_b": mx.zeros((D,), mx.float16),
        "wqkv": s(D, 3 * D), "wo": s(D, D),
        "wf1": s(D, FF), "f1b": mx.zeros((FF,), mx.float16),
        "wf2": s(FF, D), "f2b": mx.zeros((D,), mx.float16),
    } for _ in range(40)]
    V = 10502
    m.W = {"c1": s(32, 5, 5, 1), "c2": s(64, 7, 7, 32), "c3": s(192, 3, 3, 64),
           "lin": s(FLAT, D), "lin_b": mx.zeros((D,), mx.float16),
           "out": s(D, V), "out_b": mx.zeros((V,), mx.float16), "layers": m.layers}
    m.vocab = ["<blk>"] + [f"t{i}" for i in range(V - 1)]
    for t in (300, 1000, 1750):
        feats = rng.standard_normal((t, DIM)).astype(np.float32)
        t1 = (t - 5) // 5 + 1
        y3 = m._frontend(feats)[1]
        ids = m.forward(feats)
        assert len(ids) == t1, f"t={t}: ids {len(ids)} != t1 {t1}"
        print(f"  selftest t={t:5d} -> t1={t1:4d}  y3={tuple(y3.shape)}  ids={len(ids)}  ok")
        # 批窗口路径
        from .batch import BatchEngine
        B = 2
        X = np.stack([feats] * B)
        eng = BatchEngine(m, type("C", (), {"prev": 200, "blk": 1600, "post": 100})(), B,
                          compile_graph=False)
        out = eng.run([feats[k] for k in range(len(feats))], feats[0])
        assert len(out) == t1, f"batch: ids {len(out)} != t1 {t1}"
        print(f"  selftest batch t={t:5d} -> ids={len(out)}  ok")
    print("selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
