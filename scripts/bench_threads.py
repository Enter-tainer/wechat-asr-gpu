#!/usr/bin/env python3
"""TECH-42 线程扩展性基准：同一批音频，不同 WTL_THREADS 下测解码耗时 / RTF / 峰值 RSS。

用法: python3 bench_threads.py 4 6 8 12
输出: 终端表格 + bench-out/threads.json
"""
import json
import os
import statistics
import subprocess
import sys
import wave

HOME = os.path.expanduser("~")
BIN = f"{HOME}/projects/wtl-bench/target/release/wtl"
MODEL = f"{HOME}/projects/voicekey/models/wtmodel"
TDIR = f"{HOME}/projects/voicekey/test"
OUT = f"{HOME}/projects/wtl-bench/bench-out"
os.makedirs(OUT, exist_ok=True)

FILES = ["bv58.wav", "bv118.wav", "real1.wav", "real2.wav", "real3.wav", "long593.wav"]
REPS = {"long593.wav": 2}
DEFAULT_REPS = 3
THREADS = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [4, 12]


def make_long():
    """bv118 拼 5 份 ≈ 593s，用于「1 小时长音频」吞吐验证"""
    path = f"{TDIR}/long593.wav"
    if os.path.exists(path):
        return
    src = wave.open(f"{TDIR}/bv118.wav", "rb")
    p = src.getparams()
    frames = b"".join(src.readframes(p.nframes) for _ in range(5))
    src.close()
    dst = wave.open(path, "wb")
    dst.setparams(p)
    dst.writeframes(frames)
    dst.close()
    print(f"made {path} ({len(frames) // 2 / 16000:.1f}s)", flush=True)


def run_one(th, fname, rep):
    tf = f"{OUT}/_time.txt"
    cmd = ["/usr/bin/time", "-f", "%e %U %S %M", "-o", tf,
           "env", f"WTL_THREADS={th}", BIN, MODEL, f"{TDIR}/{fname}"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"wtl failed: {p.stderr[-800:]}")
    wall, user, sys_, rss = open(tf).read().split()
    line = p.stdout.rstrip("\n").split("\n")[-1]
    head, _, text = line.partition("\t")
    parts = head.split()  # name audio 58.65s 3434ms
    audio = float(parts[2].rstrip("s"))
    dec_ms = float(parts[3].rstrip("ms"))
    return dict(rep=rep, audio_s=audio, decode_ms=dec_ms, wall_s=float(wall),
                cpu_s=float(user) + float(sys_), rss_kb=int(rss), text=text)


def main():
    make_long()
    res = {}
    for th in THREADS:
        for f in FILES:
            rows = [run_one(th, f, i) for i in range(REPS.get(f, DEFAULT_REPS))]
            key = f"{th}|{f}"
            res[key] = rows
            a = rows[0]["audio_s"]
            med = statistics.median(r["decode_ms"] for r in rows)
            wall = statistics.median(r["wall_s"] for r in rows)
            cpu = statistics.median(r["cpu_s"] for r in rows)
            rss = max(r["rss_kb"] for r in rows) // 1024
            print(f"T={th:2d} {f:13s} audio={a:7.2f}s med_dec={med:8.1f}ms "
                  f"RTF={med / 1000 / a:6.4f} wall={wall:7.2f}s cpu={cpu:7.2f}s "
                  f"par_eff={cpu / wall / th:5.1%} rss={rss:4d}MB", flush=True)
        # 中断点：每档线程跑完落盘一次
        with open(f"{OUT}/threads.json", "w") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)

    print("\n== 相对 T=4 的加速比（按 med decode_ms）==")
    print(f"{'file':13s} " + " ".join(f"T={th:<7d}" for th in THREADS))
    for f in FILES:
        base = statistics.median(r["decode_ms"] for r in res[f"4|{f}"]) if f"4|{f}" in res else None
        cells = []
        for th in THREADS:
            med = statistics.median(r["decode_ms"] for r in res[f"{th}|{f}"])
            cells.append(f"{base / med:6.2f}x" if base else f"{med:7.1f}")
        print(f"{f:13s} " + " ".join(f"{c:>8s}" for c in cells))

    print("\n== 转录一致性（不同线程数文本是否完全相同）==")
    for f in FILES:
        texts = {th: res[f"{th}|{f}"][0]["text"] for th in THREADS}
        print(f"{f:13s} {'SAME' if len(set(texts.values())) == 1 else 'DIFF!!'}  len={len(texts[THREADS[0]])}")


if __name__ == "__main__":
    main()
