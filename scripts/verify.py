#!/usr/bin/env python3
"""最终验收：批模式（一个进程）+ GPU vs CPU(T=8) + 字符错误率 CER。

用法: python3 scripts/verify.py [blk] [reps]
"""
import json
import os
import statistics
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
BIN = "./target/release/wasr"
MODEL = f"{HOME}/projects/voicekey/models/wtmodel"
T1 = f"{HOME}/projects/voicekey/test"
T2 = f"{HOME}/projects/wtl-probe/test"
BLK = sys.argv[1] if len(sys.argv) > 1 else "1600"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
CASES = [("real1", f"{T1}/real1.wav"), ("real2", f"{T1}/real2.wav"), ("real3", f"{T1}/real3.wav"),
         ("bv58", f"{T1}/bv58.wav"), ("bv118", f"{T1}/bv118.wav"), ("long593", f"{T2}/long593.wav")]


def run(files, gpu):
    env = dict(os.environ)
    for k in ("WTL_GPU", "WTL_THREADS", "WTL_BLK"):
        env.pop(k, None)
    if gpu:
        env["WTL_GPU"] = "1"
    else:
        env["WTL_THREADS"] = "8"
    t0 = time.time()
    p = subprocess.run([BIN, MODEL, "--gpu" if gpu else "--cpu", "--blk", BLK] + files,
                       capture_output=True, text=True, env=env)
    wall = time.time() - t0
    if p.returncode != 0:
        raise SystemExit(p.stderr[-800:])
    out = {}
    for line in p.stdout.strip().split("\n"):
        f = line.split("\t")
        if len(f) == 5:
            out[f[0]] = dict(audio=float(f[1]), ms=float(f[2]), rtf=float(f[3]), text=f[4])
    return out, wall, p.stderr


def cer(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return dp[m] / max(n, 1)


def main():
    files = [p for _, p in CASES if os.path.exists(p)]
    names = [n for n, p in CASES if os.path.exists(p)]
    gpu, gw, gerr = run(files, True)
    cpu, cw, _ = run(files, False)
    # GPU 再跑 REPS-1 次，取中位（算法缓存已在第一次建立）
    best = {n: [] for n in names}
    for _ in range(max(0, REPS - 1)):
        g2, _, _ = run(files, True)
        for n in names:
            if n in g2:
                best[n].append(g2[n]["ms"])
    print(f"blk={BLK}  reps={REPS}\n")
    print(f"{'音频':9s}{'时长s':>8s}{'CPU-T8 ms':>11s}{'GPU ms':>9s}{'加速':>8s}{'GPU RTF':>9s}{'CER':>8s}")
    rows = {}
    for n in names:
        c, g = cpu.get(n), gpu.get(n)
        if not c or not g:
            continue
        med = statistics.median([g["ms"]] + best[n]) if best[n] else g["ms"]
        e = cer(c["text"], g["text"])
        rows[n] = dict(audio=c["audio"], cpu_ms=c["ms"], gpu_ms=med, speedup=c["ms"] / med,
                       gpu_rtf=med / 1000 / c["audio"], cer=e, text=g["text"], cpu_text=c["text"])
        print(f"{n:9s}{c['audio']:8.2f}{c['ms']:11.0f}{med:9.0f}{c['ms']/med:7.1f}x"
              f"{med/1000/c['audio']:9.5f}{e:8.4f}")
    ta = sum(v["audio"] for v in rows.values())
    print(f"\n批模式（1 进程 {len(rows)} 条，总音频 {ta:.1f}s）")
    print(f"  CPU T=8 墙钟 {cw:.2f}s → {ta/cw:.1f}x 实时")
    print(f"  GPU     墙钟 {gw:.2f}s → {ta/gw:.1f}x 实时（含首次 warmup）")
    print(f"  1 小时音频推算: CPU {3600*sum(v['cpu_ms'] for v in rows.values())/1e3/ta/60:.1f} 分钟"
          f" / GPU {3600*sum(v['gpu_ms'] for v in rows.values())/1e3/ta:.0f} 秒")
    json.dump(dict(blk=BLK, files=rows, cpu_wall=cw, gpu_wall=gw), open("/tmp/verify_result.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
