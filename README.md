# wechat-asr-gpu

微信输入法**离线 ASR 引擎**（上游 `voicekey/crates/wtlocal`，模型容器 `.xnet`）的纯 Rust + CUDA 复刻，面向**批模式批量转写**：给一堆 wav，全部扔给 GPU 出文本。

- **1 小时音频单文件：3577.9s → 解码 2.77s（RTF 0.0008），端到端墙钟 3.27s = 1095x 实时**，峰值 RSS 869 MB
- 单流长音频：GPU 比 CPU（8 线程）快 **35~44x**（音频越长越高），GPU 侧 RTF ≈ 0.0008（≈1100x 实时）
- 批量吞吐：796s 音频 **2.80s 墙钟**（≈**284x 实时**，含首次预热与模型加载）
- 正确性：int8 矩阵乘的 int32 累加与 CPU 参考实现**逐元素完全相同**；转录文本 CER 0~2.6%（差异全是同音字/单字级）
- 无 GPU 也能跑：CPU 后端（int8 + AVX2/NEON + rayon）保留，同一份代码 `--cpu` 切换

## 实测（RTX 5060 Ti 16GB，12 核 Intel；同一进程批处理 6 条，`--blk 1600`）

| 音频 | 时长 | CPU (8T) | GPU | 加速 | GPU RTF | CER vs CPU |
|---|---:|---:|---:|---:|---:|---:|
| real1 | 8.75s | 221ms | 90ms¹ | 2.4x | 0.0103 | **0.0%（逐字一致）** |
| real2 | 8.31s | 179ms | 7ms | **25.6x** | 0.0008 | 2.9% |
| real3 | 8.88s | 196ms | 7ms | **28.0x** | 0.0008 | **0.0%（逐字一致）** |
| bv58 | 58.65s | 1821ms | 53ms | **34.4x** | 0.0009 | **0.0%（逐字一致）** |
| bv118 | 118.61s | 3930ms | 106ms | **37.3x** | 0.0009 | 1.9% |
| long593 | 593.04s | 20050ms | 455ms | **44.1x** | 0.0008 | 1.2% |

批聚合（一个进程、6 条、总音频 796.2s）：CPU 墙钟 26.6s（30x 实时）→ **GPU 2.80s（284x 实时）**；**1 小时音频约 3 秒**。
¹ 第一条含一次性预热（CUDA 上下文 + 130MB 权重上传 ≈2s）。cuBLASLt 算法选择**持久化到 `~/.cache/wasr/lt_algo.bin`**，同机器/同 CUDA 版本只搜一次。

## 极限：1 小时单文件

| 音频 | 时长 | 解码 | 墙钟（含 ~1.6s 加载） | 速度 | GPU RTF | 峰值 RSS |
|---|---:|---:|---:|---:|---:|---:|
| bv58 ×61 拼接 | 3577.9s（59.6 min） | **2.77s** | **3.27s** | **1095x 实时** | 0.0008 | 869 MB |

- 单流、单文件、`--blk 1600` → 224 个解码窗口；>200s 的音频自动走**整窗 CUDA graph replay**，所以窗口越多越不吃 host 开销。
- 峰值 RSS 由 `/usr/bin/time -v` 实测（868,200 kB）；解码纯耗时 2.77s 对应 ≈1290x 实时（RTF 0.0008）。
- 顺带说明：这个场景下 GPU 上的瓶颈已经不是算力，而是**权重带宽**（每窗口要过一遍 126MB 权重，L2 只有 32MB）。

## 架构

```
wav ──► fbank (Rust, CPU)  log-mel 39+1，25ms 窗 / 10ms 帧移，滚动均值归一化
     ──► 卷积前端 (CUDA)    5×5 1→32 / 7×7 32→64 stride5 / 3×3 64→192 freq stride2
     ──► 40 × Transformer (CUDA)   pre-norm, D=512, 8 头, FF=2048, GELU
             norm+quant | q,k,v GEMM(int8) | attention | o GEMM | add+norm+quant | f1 GEMM | gelu+quant | f2 GEMM | add
     ──► output GEMM (int8, 词表 10502) + argmax (GPU) ──► token id
     ──► CTC 折叠 / 标点整理 (Rust) ──► 文本
```

- **Rust 负责引擎外壳**：fbank、`.xnet` 容器解析、权重 blob 打包（256B 对齐）、分块解码、CTC 折叠、CLI。约 1.5k 行。
- **CUDA 负责全部计算**，编成**单个** `libwasr_gpu.so`（`csrc/` 下 4 个 `.cu`，符号前缀 `wp_*`/`wk_*` 互不冲突）：
  - `wtlgpu.cu` — int8 GEMM（cuBLASLt，int32 累加）+ 权重自动对齐补齐 + CUDA graph 封装
  - `wtl_kern.cu` — 卷积前端、norm+quant、add+norm+quant、gelu+quant、argmax（融合算子，减少 kernel 数与显存往返）
  - `wk_attention_fast.cu` / `wk_attn_api.cu` — attention（cuBLAS batched sgemm + 连续内存 softmax；另有 tiled/fp32 实现可 A/B）
- 所有算子跑同一条 CUDA stream，发起顺序即语义顺序；每个解码窗口只在取 token id 前同步一次。

## 构建

需要：Rust ≥ 1.75、CUDA Toolkit ≥ 12（含 `nvcc`）、以及能跑 cuBLASLt 的 NVIDIA GPU（sm_80+；本机实测 sm_120）。

```bash
cargo build --release
# 可选环境变量：
#   NVCC=/usr/local/cuda/bin/nvcc       nvcc 路径
#   WTL_CUDA_ARCH=sm_120                -arch（默认 native）
#   WTL_SKIP_CUDA=1                     只编 Rust（调试语法用，GPU 不可用）
```

`build.rs` 会调 nvcc 把 `csrc/*.cu` 编成一个 `libwasr_gpu.so` 并链进可执行文件（rpath 已内嵌）。

## 准备模型

仓库**不包含任何模型权重**（权重来自微信输入法官方语音包，版权归腾讯，请自行合法获取）：

```bash
# 官方语音包（.apk）→ 模型目录（xnet + dict.decoder.utf8.txt）
./target/release/wasr --unpack /path/to/voicepkg.apk ./models/wtmodel
```

## 使用

```bash
# 批量转写（GPU，Tab 分隔输出：名字 音频秒 解码毫秒 RTF 文本）
./target/release/wasr ./models/wtmodel --gpu a.wav b.wav c.wav

# 顺带写 JSON 和逐条 txt
./target/release/wasr ./models/wtmodel --gpu --json out.json --txt out_txt/ *.wav

# CPU 对照（8 线程）
./target/release/wasr ./models/wtmodel --cpu --threads 8 *.wav
```

`--blk`（解码块长，特征帧/10ms）默认 **1600**（批模式最优，重算放大 1.22x）；流式场景才需要 400。

## 正确性

- int8 矩阵乘：cuBLASLt 的 int32 累加结果与 CPU 参考实现**逐元素完全相同**（对拍 mismatch=0）。
- 端到端：与 CPU 后端转录对比，`real2`/`bv58` 逐字一致，其余 CER 0~2.6%，差异全部是同音字/单字级（场↔城、养↔羊、香↔乡、魂↔闺），长音频无级联崩塌。
- 差异根因：fp32 求和顺序不同（attention 走 cuBLAS，误差 1.2e-7；卷积前端/softmax 用并行归约），量级与 fp32 舍入同级。需要"逐字一致"时可切精确顺序内核（慢约 25%）。
- 验收脚本：`scripts/verify_gpu.py`（逐字/字符错误率 + 加速比）、`scripts/final_bench.py`（批模式吞吐）。

## 目录

```
csrc/       CUDA 源码（编成单个 libwasr_gpu.so）
src/        Rust：fbank / xnet / model / stream / kernels(CPU) / gpu(FFI) / bin/wasr
scripts/    基准与验收脚本
docs/       TECH-42 完整实测报告（线程扩展、逐阶段拆解、路线选型、nsys 数据）
```

## 三处已落地的重要优化

- **整窗 CUDA graph**：把「linear + 40 层 + 输出层 + argmax」约 806 个 kernel 捕获成一张图，同 t1 的窗口直接 replay（host 入队 3ms/窗口 → ~1.5µs）。
  - 捕获前会先用几十 µs 把所有 (k,n,t) 形状喂给 cuBLASLt + 触发 attention 的 cublas handle 创建，避免 capture 里做非法初始化。
  - **只对长音频启用**（`Stream` 里按总特征帧数 ≥20000 即 200s 判断）：短音频捕获/实例化开销大于省下的时间。
  - `WTL_GRAPH=1/0` 强制开/关，`WTL_NO_GRAPH=1` 调试对照。实测 long593（593s）**-6%**，且与逐 kernel 输出逐字一致。

## 另外两处

- **fbank 并行化**：批模式整段喂入时按帧并行算 FFT/mel（滚动均值仍按原顺序累加，数值与单线程逐位一致），58.65s 音频的 fbank 从 29ms 降到 ~4ms；`WTL_FBANK_SERIAL=1` 可回退串行做对拍。
- **cuBLASLt 算法选择磁盘缓存**：首次按形状实测候选算子的耗时（每形状几十 ms），选中结果写入 `$WTL_ALGO_CACHE`（默认 `~/.cache/wasr/lt_algo.bin`，记录 GPU 名 / driver / CUDA 版本做失效保护）；之后同一机器直接复用，跳过整段搜索。
  - `WTL_ALGO_CACHE=/path/to/file` 指定位置；`=off` 关闭；`WTL_ALGO_CACHE_LOG=1` 打印命中/写回。
  - 若缓存的算法在实际调用中报错，会自动删条目并重新搜索。

## 性能现状与后续

单窗口（t=150）耗时约 7ms：卷积前端 0.02ms + 40 层 7ms（attention 0.75、conv2+conv3 1.3、norm/quant 类 1.3、int8 GEMM 0.9、GEMM host 入队 1.28、epilogue+gelu 1.0，单位 ms）。

按阿姆达尔排序的剩余优化（都还没做）：

1. 整窗口 CUDA graph（host 入队 1.28ms → ~1.5µs，-12%；库已有 capture API）
2. conv2/conv3 的 shared-memory tiling（-9%）
3. norm/quant 并行归约（-9%）
4. TileLang/CUTLASS FMHA 版 attention（现状 18.8µs/次，目标 <10µs，-5%）

megakernel 不做：graph 已吃掉 host 开销，剩下都是真实计算。

## License

MIT。模型权重、`.xnet` 容器格式与 `dict.decoder.utf8.txt` 均来自微信输入法官方语音包，版权归腾讯所有，本仓库不包含、不分发。
