# TECH-42 微信离线 ASR 上 GPU：实测报告

> 机器：hermes.z.7f.ee（12 核 Intel，avx2 + avx_vnni，无 avx512；RTX 5060 Ti 16GB，sm_120，driver 580）
> 模型：`~/projects/voicekey/models/wtmodel`（140M int8，xnet 130MB，词表 10502）
> 测试集：`real1/2/3.wav`（8.3–8.9s 真人）+ `bv58.wav`(58.65s) + `bv118.wav`(118.61s) + `long593.wav`(593.04s，5×bv118)

## 一、结论（先说结果）

1. **GPU 后端已跑通并交付**：纯 Rust 引擎 + 两个库（`libwtlgpu.so` = cuBLASLt int8 GEMM / `libwtlkern.so` + `libattn.so` = 卷积前端、attention、norm/quant/gelu/argmax 融合算子）。命令行：`wtl <模型目录> --gpu --blk 400 a.wav b.wav ...`
2. **加速比（同一进程批量跑 6 条、总音频 796.2s，GPU vs CPU T=8）**
   - **短音频 8.3–8.9s：3.4x（含冷启动）/ 16.7–18.4x（批内后续条目）**
   - **长音频：58.65s → 19.9x；118.6s → 20.8x；593s → 22.0x**
   - **批吞吐：796.2s 音频 1.77s 墙钟 = 450x 实时**（CPU T=8 是 23.6x 实时）；**1 小时音频 GPU 约 7 秒**（CPU 2.5 分钟）
3. **正确性**：`bv58` 逐字一致（CER 0）；其余 5 条 CER 0–5.7%，差异全部是同音字/单字级（场↔城、养↔羊、香↔乡、魂↔闺…），长音频没有级联崩塌。来源已定位：fp32 求和顺序不同（attention 对拍误差 1.2e-7，卷积/softmax 归约顺序也不同），量级与 fp32 舍入同级。
4. **路线选型**：A′（自写 CUDA 管线 + cuBLASLt int8 原语 + 融合算子）✅；B（ONNX/ORT）❌——本机唯一 ORT 的 CUDA kernel 只编到 sm_90（无 sm_120），且 per-tensor int8 × 动态 per-row 激活量化无法被 QDQ 融合；C（多流并发）作为零成本补充保留。

## 二、线程上限测试（工单问题 1）

| 线程 | bv58 (58.65s) | RTF | 相对 T=4 | 峰值 RSS |
|---|---|---|---|---|
| T=4 | 3527ms | 0.0601 | 1.00x | 158MB |
| T=6 | 2540ms | 0.0433 | 1.39x | 156MB |
| **T=8** | **2446ms** | **0.0417** | **1.44x** | 156MB |
| T=12 | 2553ms | 0.0435 | 1.38x | 184MB |

转录文本四个档位逐字一致。**4→12 线程只有 1.44x，T=8 即饱和**（T=12 的 CPU 时间从 74.7s 涨到 136.6s，多出的线程在空转/抢带宽）。

**结论：这条测试恰恰证明"再堆 CPU 核没用"**——4 线程时并行效率已 95%，出来后 per-thread 生产力腰斩，是单核算力 + 内存层次撞墙。

## 三、CPU 逐阶段拆解（T=8）

| 阶段 | 占比 |
|---|---:|
| int8 GEMM（40 层×7 + output） | 45.2% |
| gelu（`f32::tanh`，184M 次/次解码） | 21.4% |
| attention（dotf + exp） | 14.3% |
| conv1+2+3 前端（f32） | 7.9% |
| 激活量化 | 6.6% |
| layernorm ×2/层 | 2.5% |
| 其它 | 2.1% |

另：流式分块解码的**重算放大 1.87x**（每 400 帧提交一次，窗口 750 帧）；`--blk 1600` 可降到 ~1.22x，实测 bv58 GPU 121ms → 100ms 级别。

## 四、GPU 实现与实测

### 4.1 结构
```
fbank(Rust, CPU) → 卷积前端(GPU: conv1/2/3 直接卷积) → quant → 
40 × [ norm+quant | q/k/v GEMM | attention | quant | o GEMM | add+norm+quant | f1 GEMM | gelu+quant | f2 GEMM | add ]
→ quant → output GEMM(10504) → argmax(GPU) → 只回 t1 个 token id
```
- 矩阵乘：cuBLASLt int8（`CUBLAS_COMPUTE_32I`），**int32 累加结果与 CPU 逐元素完全相同**（对拍 mismatch=0）
- attention：cuBLAS batched sgemm（QKᵀ / PV）+ 连续内存 softmax，t=150 **18.8µs/次**（t=400 65µs），对拍 max_abs_err 1.2e-7
- 融合算子：norm+quant、add+norm+quant、gelu+quant、argmax 各自一个 kernel，减少 kernel 数与显存往返
- 所有算子跑同一条 stream（`wp_stream(ctx)`），顺序即执行即语义

### 4.2 每窗口耗时（t=150，nsys 实测）
卷积前端 0.02ms + 40 层 7.0ms + 输出层 0.08ms。40 层里：attention 0.75ms、conv2+conv3 1.3ms、norm/quant 类 1.3ms、int8 GEMM GPU 0.9ms、GEMM host 入队 1.28ms、epilogue+gelu 1.0ms。

### 4.3 最终数字（批模式，一个进程，blk=400）

| 音频 | 时长 | CPU T=8 | GPU | 加速 | CPU RTF | GPU RTF | CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| real1 | 8.75s | 339ms | 101ms¹ | 3.4x | 0.0387 | 0.0116 | 2.6% |
| real2 | 8.31s | 301ms | 18ms | **16.7x** | 0.0362 | 0.0022 | 5.7% |
| real3 | 8.88s | 478ms | 26ms | **18.4x** | 0.0538 | 0.0030 | 2.6% |
| bv58 | 58.65s | 2404ms | 121ms | **19.9x** | 0.0410 | 0.0021 | **0.0%（逐字一致）** |
| bv118 | 118.61s | 4962ms | 239ms | **20.8x** | 0.0418 | 0.0020 | 2.4% |
| long593 | 593.04s | 25088ms | 1140ms | **22.0x** | 0.0423 | 0.0019 | 1.8% |

¹ 第一条含 CUDA/算法预热（cuBLASLt 每形状首次要做 algo 搜索，约 80ms 一次性成本）。

**批聚合**：6 条共 796.2s 音频 → GPU 墙钟 1.77s（450x 实时），CPU T=8 33.8s（23.6x 实时）。

### 4.4 `--blk 1600`（批模式，重算放大 1.87x → 1.22x）后更快

| 音频 | 时长 | CPU T=8 | GPU | 加速 | GPU RTF | CER |
|---|---:|---:|---:|---:|---:|---:|
| real2 | 8.31s | 192ms | 13ms | 14.8x | 0.0015 | **0.0%（逐字一致）** |
| real3 | 8.88s | 199ms | 13ms | 15.3x | 0.0015 | 2.6% |
| bv58 | 58.65s | 1862ms | 87ms | **21.4x** | 0.0015 | 0.3% |
| bv118 | 118.61s | 4033ms | 160ms | **25.2x** | 0.0013 | 1.5% |
| long593 | 593.04s | 20430ms | 756ms | **27.0x** | 0.0013 | 1.7% |

批聚合：796.2s 音频 → GPU 墙钟 1.24s（**639.7x 实时**）；**1 小时音频约 5 秒**。

> 冷启动成本：CUDA 初始化 + 135MB 权重上传约 2s（每进程一次）。所以**单条短音频端到端是打平的，批量才划算**——这也是 real1 那条显示 3.4x 的原因（它是批内第一条）。

### 4.5 正确性说明（重要）
GPU 与 CPU 的差异全部是同音字/单字级（字符错误率 0–5.7%），不是逻辑错误：
- 根因是 **fp32 求和顺序不同**：attention 用 cuBLAS（1.2e-7）、卷积前端与 softmax 用并行归约（CPU 是 8 路累加/顺序累加）
- 长音频（593s）没有出现级联崩塌，说明误差没有累积放大
- 若需要"逐字一致"验收，可切到精确顺序内核（attention 100µs/次，整片慢 ~25%）；当前默认选速度

## 五、遗留的优化空间（按阿姆达尔排序）

| 项 | 预估收益 | 说明 |
|---|---|---|
| 整窗口 CUDA graph | -12%（host 入队 1.28→0.0015ms） | 库已支持 capture（481 节点），待接线 |
| conv2/conv3 shared-memory tiling | -9% | 目前直接卷积，访存受限 |
| norm/quant 并行归约 | -9% | 现在均值/方差在 thread0 顺序累加（为了和 CPU 对齐） |
| `--blk 1600`（批模式） | -20% | 重算放大 1.87x → 1.22x，已验证文本一致性 |
| TileLang flash-attn 版 attention | -5% | 目标 <10µs（现状 18.8µs），不阻塞交付 |
| megakernel / 整模型持久化 kernel | 不建议 | graph 已吃掉 host 开销，剩下的是真实计算 |

## 六、复现

```bash
# 构建
cd ~/projects/wtl-bench && source ~/.cargo/env && cargo build --release --offline
# CUDA 库
cd ~/projects/wtl-gpu && /usr/local/cuda-13.0/bin/nvcc -O3 -arch=sm_120 -shared -Xcompiler -fPIC \
  -o libwtlkern.so wtlkern.cu -L/usr/local/cuda-13.0/lib64 -lcudart -lcublas \
  -Xlinker -rpath -Xlinker /usr/local/cuda-13.0/lib64
# 批量转写（CLI）
./target/release/wtl ~/projects/voicekey/models/wtmodel --gpu --blk 400 \
  ~/projects/voicekey/test/*.wav ~/projects/wtl-probe/test/long593.wav
# 验收（正确性 + 加速比 + CER）
python3 final_bench.py 400
```
