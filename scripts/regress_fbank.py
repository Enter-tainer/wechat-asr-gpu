#!/usr/bin/env python3
"""定位差异来源：fbank 并行化是否改变数值；GPU 文本变化是否只是编译/后端差异。

对比四份文本：
  old_cpu / old_gpu  ← wtl-bench/bench-out/final_bench.json（优化前，三库构建）
  new_cpu / new_gpu  ← 当前 repo（fbank 并行化 + 单库构建）
"""
import json
import os
import subprocess

HOME = os.path.expanduser("~")
BIN = "./target/release/wasr"
M = f"{HOME}/projects/voicekey/models/wtmodel"
OLD = json.load(open(f"{HOME}/projects/wtl-bench/bench-out/final_bench.json"))["files"]
CASES = [("bv58", f"{HOME}/projects/voicekey/test/bv58.wav"),
         ("bv118", f"{HOME}/projects/voicekey/test/bv118.wav"),
         ("long593", f"{HOME}/projects/wtl-probe/test/long593.wav")]


def run(path, gpu):
    env = dict(os.environ)
    if gpu:
        env["WTL_GPU"] = "1"
    else:
        env.pop("WTL_GPU", None)
        env["WTL_THREADS"] = "8"
    p = subprocess.run([BIN, M, "--gpu" if gpu else "--cpu", path], capture_output=True, text=True, env=env)
    return [l for l in p.stdout.strip().split("\n") if l][-1].split("\t")[4]


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


print(f"{'file':9s}{'old_cpu==new_cpu':>18s}{'old_gpu==new_gpu':>18s}{'new_cpu vs new_gpu':>20s}")
for name, path in CASES:
    old = OLD[name]
    new_cpu, new_gpu = run(path, False), run(path, True)
    print(f"{name:9s}"
          f"{('逐字一致' if old['cpu_text'] == new_cpu else 'CER %.4f' % cer(old['cpu_text'], new_cpu)):>18s}"
          f"{('逐字一致' if old['gpu_text'] == new_gpu else 'CER %.4f' % cer(old['gpu_text'], new_gpu)):>18s}"
          f"{('逐字一致' if new_cpu == new_gpu else 'CER %.4f' % cer(new_cpu, new_gpu)):>20s}")
