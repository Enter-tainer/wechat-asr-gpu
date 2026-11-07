#!/usr/bin/env python3
"""最终验收：批模式（一个进程处理多条音频）+ 正确性（字符错误率 CER）+ 短/长音频加速比。

用法: python3 final_bench.py [blk]
输出: 控制台表 + final_bench.json
"""
import json
import os
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
BIN = f"{HOME}/projects/wtl-bench/target/release/wtl"
MODEL = f"{HOME}/projects/voicekey/models/wtmodel"
T1 = f"{HOME}/projects/voicekey/test"
T2 = f"{HOME}/projects/wtl-probe/test"
OUT = f"{HOME}/projects/wtl-bench/bench-out"
os.makedirs(OUT, exist_ok=True)
BLK = sys.argv[1] if len(sys.argv) > 1 else "400"

SHORT = [f"{T1}/real1.wav", f"{T1}/real2.wav", f"{T1}/real3.wav"]
LONG = [f"{T1}/bv58.wav", f"{T1}/bv118.wav", f"{T2}/long593.wav"]
ALL = SHORT + LONG


def run_batch(env_extra, files):
    env = dict(os.environ)
    for k in ("WTL_GPU", "WTL_THREADS", "WTL_BLK"):
        env.pop(k, None)
    env.update(env_extra)
    t0 = time.time()
    p = subprocess.run([BIN, MODEL, "--blk", BLK] + files, capture_output=True, text=True, env=env)
    wall = time.time() - t0
    if p.returncode != 0:
        raise SystemExit(f"失败: {p.stderr[-800:]}")
    rows = {}
    for line in p.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) == 5:
            rows[parts[0]] = dict(audio=float(parts[1]), ms=float(parts[2]), rtf=float(parts[3]), text=parts[4])
    return rows, wall, p.stderr


def cer(ref: str, hyp: str) -> float:
    """字符错误率（编辑距离 / 参考长度）"""
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
    print(f"解码块长 blk={BLK}\n")
    cpu, cpu_wall, _ = run_batch({"WTL_THREADS": "8"}, ALL)
    gpu, gpu_wall, gpu_err = run_batch({"WTL_GPU": "1"}, ALL)

    print(f"{'音频':12s}{'时长s':>8s}{'CPU-T8 ms':>11s}{'GPU ms':>9s}{'加速':>7s}"
          f"{'CPU RTF':>9s}{'GPU RTF':>9s}{'CER':>8s}{'文本':>6s}")
    out = {}
    for f in ALL:
        name = os.path.splitext(os.path.basename(f))[0]
        c, g = cpu.get(name), gpu.get(name)
        if not c or not g:
            print(f"缺失 {name}")
            continue
        e = cer(c["text"], g["text"])
        same = c["text"] == g["text"]
        out[name] = dict(audio=c["audio"], cpu_ms=c["ms"], gpu_ms=g["ms"], speedup=c["ms"] / g["ms"],
                         cpu_rtf=c["rtf"], gpu_rtf=g["rtf"], cer=e, same=same,
                         cpu_text=c["text"], gpu_text=g["text"])
        print(f"{name:12s}{c['audio']:8.2f}{c['ms']:11.0f}{g['ms']:9.0f}{c['ms']/g['ms']:6.1f}x"
              f"{c['rtf']:9.5f}{g['rtf']:9.5f}{e:8.4f}{'一致' if same else '差异':>6s}")

    ca = sum(v["audio"] for v in out.values())
    print(f"\n批模式（一个进程跑 {len(out)} 条，总音频 {ca:.1f}s）")
    print(f"  CPU T=8: 解码合计 {sum(v['cpu_ms'] for v in out.values())/1e3:.1f}s, 墙钟 {cpu_wall:.1f}s, 吞吐 {ca/cpu_wall:.1f}x 实时")
    print(f"  GPU:     解码合计 {sum(v['gpu_ms'] for v in out.values())/1e3:.1f}s, 墙钟 {gpu_wall:.1f}s, 吞吐 {ca/gpu_wall:.1f}x 实时")
    print(f"  端到端批吞吐加速: {cpu_wall/gpu_wall:.1f}x")
    print(f"  1 小时音频推算: CPU {3600*sum(v['cpu_ms'] for v in out.values())/1e3/ca/60:.1f} 分钟"
          f" / GPU {3600*sum(v['gpu_ms'] for v in out.values())/1e3/ca:.0f} 秒")
    if gpu_err.strip():
        print("\nGPU stderr（前 5 行）:")
        print("\n".join(gpu_err.strip().split("\n")[:5]))
    with open(f"{OUT}/final_bench.json", "w") as fh:
        json.dump(dict(blk=BLK, files=out, cpu_wall=cpu_wall, gpu_wall=gpu_wall), fh, ensure_ascii=False, indent=1)
    print(f"\nJSON: {OUT}/final_bench.json")


if __name__ == "__main__":
    main()
