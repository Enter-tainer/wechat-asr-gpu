#!/usr/bin/env python3
"""TECH-42 验收：GPU 后端正确性 + 短/长音频加速倍数。

正确性判据：同一音频、同一组解码参数下，GPU 转录文本必须与 CPU 参考实现**逐字相同**。
每个 (文件, 配置) 都单开一个进程（库的 (k,n,t) 算法缓存只有 24 条，跨文件会打满）。

用法: python3 verify_gpu.py [--reps N]
输出: 表格 + verify_gpu.json
"""
import json
import os
import statistics
import subprocess
import sys

HOME = os.path.expanduser("~")
BIN = f"{HOME}/projects/wtl-bench/target/release/wtl"
MODEL = f"{HOME}/projects/voicekey/models/wtmodel"
T1 = f"{HOME}/projects/voicekey/test"
T2 = f"{HOME}/projects/wtl-probe/test"
OUT = f"{HOME}/projects/wtl-bench/bench-out"
os.makedirs(OUT, exist_ok=True)

CASES = [
    ("real1.wav", T1),
    ("real2.wav", T1),
    ("real3.wav", T1),
    ("bv58.wav", T1),
    ("bv118.wav", T1),
    ("long593.wav", T2),
]

# (tag, env, blk)
CONFIGS = [
    ("CPU-T4-blk400", {"WTL_THREADS": "4"}, "400"),
    ("CPU-T8-blk400", {"WTL_THREADS": "8"}, "400"),
    ("GPU-blk400", {"WTL_GPU": "1"}, "400"),
    ("CPU-T8-blk1600", {"WTL_THREADS": "8"}, "1600"),
    ("GPU-blk1600", {"WTL_GPU": "1"}, "1600"),
]
REPS = 2


def run(env_extra, fname, tdir, blk):
    env = dict(os.environ)
    for k in ("WTL_GPU", "WTL_THREADS", "WTL_BLK"):
        env.pop(k, None)
    env.update(env_extra)
    p = subprocess.run([BIN, MODEL, "--blk", blk, f"{tdir}/{fname}"], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        return None, p.stderr[-300:]
    line = [l for l in p.stdout.strip().split("\n") if l][-1]
    parts = line.split("\t")
    if len(parts) < 5:
        return None, line
    return dict(name=parts[0], audio=float(parts[1]), ms=float(parts[2]),
                rtf=float(parts[3]), text=parts[4]), None


def main():
    res = {}
    print(f"{'文件':14s}{'时长':>8s}{'配置':>16s}{'最长时间':>10s}{'RTF':>9s}")
    for fname, tdir in CASES:
        if not os.path.exists(f"{tdir}/{fname}"):
            print(f"skip {fname}")
            continue
        for tag, env, blk in CONFIGS:
            runs = []
            err = None
            for _ in range(REPS):
                r, err = run(env, fname, tdir, blk)
                if r:
                    runs.append(r)
            if not runs:
                print(f"{fname:14s}{'-':>8s}{tag:>16s}  失败: {err}")
                continue
            med = statistics.median(x["ms"] for x in runs)
            res[f"{fname}|{tag}"] = dict(audio=runs[0]["audio"], ms=med, rtf=med / 1000 / runs[0]["audio"],
                                         text=runs[0]["text"])
            print(f"{fname:14s}{runs[0]['audio']:8.2f}{tag:>16s}{med:10.0f}{med/1000/runs[0]['audio']:9.5f}", flush=True)

        # 正确性：同 blk 下 GPU 文本 == CPU 文本
        for blk, cpu_tag, gpu_tag in (("400", "CPU-T8-blk400", "GPU-blk400"),
                                      ("1600", "CPU-T8-blk1600", "GPU-blk1600")):
            c = res.get(f"{fname}|{cpu_tag}")
            g = res.get(f"{fname}|{gpu_tag}")
            if not c or not g:
                continue
            same = c["text"] == g["text"]
            n = max(len(c["text"]), len(g["text"]))
            diff = sum(1 for a, b in zip(c["text"], g["text"]) if a != b) + abs(len(c["text"]) - len(g["text"]))
            sp = c["ms"] / g["ms"]
            print(f"  └ blk={blk:4s} 正确性: {'逐字一致 ✓' if same else f'不一致 ✗ ({diff}/{n} 字符)'}"
                  f"   GPU 相对 {cpu_tag} 提速 {sp:.1f}x", flush=True)

    with open(f"{OUT}/verify_gpu.json", "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)

    print("\n== 加速倍数汇总（GPU vs CPU，中位耗时）==")
    print(f"{'音频':14s}{'时长':>8s}{'CPU T=8':>10s}{'GPU':>8s}{'加速':>8s}{'CPU RTF':>9s}{'GPU RTF':>9s}")
    for fname, _ in CASES:
        c = res.get(f"{fname}|CPU-T8-blk400")
        g = res.get(f"{fname}|GPU-blk400")
        if c and g:
            print(f"{fname:14s}{c['audio']:8.2f}{c['ms']:10.0f}{g['ms']:8.0f}"
                  f"{c['ms']/g['ms']:7.1f}x{c['rtf']:9.5f}{g['rtf']:9.5f}")


if __name__ == "__main__":
    main()
