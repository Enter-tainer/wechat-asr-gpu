# Mac / Apple Silicon 上跑这个 ASR 引擎：MLX 移植报告

> 机器：Apple M2 Pro（10 核 CPU / 16 核 GPU / 32GB / macOS 27.2 / Metal 4）
> 工具链：Rust 1.96.1、Python 3.9（Xcode）、MLX 0.29.3、numpy 2.0.2
> 对照机：hermes（12 核 Intel + RTX 5060 Ti 16GB），CUDA int8 版实测见 `TECH-42-GPU-report.md`
> 测试音频：`bv58.wav`（58.65s 真人）、`real1.wav`（8.75s 真人）

## 一、结论

1. **能做，而且值得做：MLX 版实测 345x 实时（bv58：170ms），比同机 CPU 快 12 倍，只比 RTX 5060 Ti 的
   CUDA int8 版慢 3~4 倍**（不是 10 倍——见第三节，CUDA 版自己也没吃满 int8 张量核）。
2. **选 MLX，不要手写 Metal** —— 除非验收要求"逐字一致"。理由见第二节的能力矩阵：
   CUDA 版的 91% 算力是一个 int8 GEMM，而 Apple GPU（M1–M4）**根本没有 int8 矩阵单元**，
   这条路在 Mac 上不存在，无论用 MLX 还是 Metal 都必须换成 fp16。一旦确定 fp16，
   MLX 的 `matmul` / `fast.scaled_dot_product_attention` / `mx.compile` 已经把 40 层整窗开到
   **4.0 TFLOPS（约 35% 峰值）**，手写 Metal 大概还有 1.5x 上限，但要写 1600~2300 行 MSL + 绑定。
3. **int8 那份"逐位一致"的验收契约在 Apple GPU 上只能靠手写 Metal int8 kernel 保住**
   （整数加法与顺序无关，int32 累加天然逐位一致）。fp16 路线的差异是 CER 级别的：
   实测与 CUDA 版转录 **CER ≈1.2%**，与现有 CPU↔GPU 的 0~5.7% 同量级。
4. **如果只是"想在 Mac 上更快地转写"**，0 开发量的方案是走 Tailscale 调 hermes 上的 CUDA 版
   （1100~1300x）。MLX 版的价值在于**离线、便携、单机可复现**。

## 二、Apple 侧能力矩阵（决定选型的硬事实）

| 需求（CUDA 版里的东西） | MLX | Metal / MPS | CoreML / ANE |
|---|---|---|---|
| int8×int8 → int32 GEMM | ❌ `mx.matmul` 只收浮点 | ❌ M1–M4 无 int8 矩阵指令；MPSMatrix 收 int8 但只出 fp16/fp32 且更慢 3x；M5+ 的 Metal 4 TensorOps `matmul2d` 才有原生 int8 | ❌ 静态 per-tensor 量化，不支持 per-row 动态 |
| weight-only int8（反量化后 GEMM） | ⚠️ `quantized_matmul` 可用但**实测更慢**（0.76~0.89x） | — | — |
| fp16 GEMM | ✅ 实测 1.8~4.7 TFLOPS | ✅ 手写 simdgroup 可到 5~6 TFLOPS 量级 | — |
| 融合 epilogue（反量化+GELU+重量化） | ✅ `mx.compile` 自动融合逐元素链 | ✅ 自己写 kernel 最自由 | ✅ 但只能静态图 |
| attention | ✅ `mx.fast.scaled_dot_product_attention`（实测 4.26 TFLOPS @t=390） | ✅ 需手写 flash-attn | ⚠️ 需固定形状 |
| CUDA graph（整窗 ~800 kernel） | ✅ `mx.compile`（实测 -29%） | ✅ `MTLIndirectCommandBuffer` 可做到 µs 级 replay | — |
| 单流顺序语义 / 显存管理 | ✅ 惰性图 + 统一内存，零拷贝 | ✅ MTLBuffer 统一内存 | ✅ |

**一句话**：Mac 上"int8 推理"这条路是断的，只能 fp16/bf16；fp16 之后 MLX 是最短路径。

## 三、实测数据

### 3.1 三个后端的端到端对照（同一批音频）

| 后端 | bv58（58.65s） | 倍实时 | 1 小时音频 | CER |
|---|---:|---:|---:|---:|
| MLX fp16（本报告，M2 Pro 16 核 GPU，`--mlx --batch 4`） | **141 ms** | **416x** | ≈9 s | 1.2%（vs CUDA） |
| Rust CPU（M2 Pro，`--cpu`，标量回退） | 3306 ms | 17.7x | ≈3.4 min | 0（参考实现） |
| CUDA int8（RTX 5060 Ti） | 53 ms | 1107x | ≈3 s | 0~1.7%（vs CPU） |

> Mac 只比 5060 Ti 慢 **3.2x**，而两边的理论算力差 ~16x（fp16 11.5 TFLOPS vs int8 180 TOPS）。
> 原因是 CUDA 版在 t1=390 这种小窗口下也只跑到 27.7 TOPS（峰值的 17%），瓶颈是 kernel 数×固定开销
> 与每窗 126MB 权重流；小窗口把两边的差距拉平了。**这也说明 Mac 侧还有一点点空间，但不多。**

### 3.2 CPU int8 微基准（M2 Pro 实测，`bench_cpu_neon.rs`）

复刻 `src/kernels.rs` 的 `dot2x4`（vdotq_s32）结构，t=1600 / k=512 / n=2048：

| 线程 | GMAC/s | 4~10 线程的扩展 |
|---|---:|---:|
| 1 | 118.7 | — |
| 4 | 346.4 | 2.9x |
| 8 | 530.0 | 4.5x |
| 10 | 658.9 | 5.6x |

即 M2 Pro 的 CPU int8 路径大约 15~28x 实时。**注意：`vdotq_s32` 在 stable Rust 上是 unstable
（`stdarch_neon_dotprod`，E0658），这也是为什么 Rust 版在 Mac 上今天还跑不起来（见第五节）。**

### 3.3 MLX 单 GEMM / 整窗吞吐（M2 Pro）

| 形状 | 时间 | TFLOPS |
|---|---:|---:|
| (1600,512)@(512,2048) | 0.89 ms | 3.76 |
| (1600,2048)@(2048,512) | 1.43 ms | 2.34 |
| (6400,512)@(512,2048)（B=4 批） | 2.88 ms | 4.66 |
| 40 层整窗（T=1600 输出帧） | 153 ms | 4.0（含 attention） |

### 3.4 数值漂移

同一次 int8 量化，三条路径对比（FF up，t=1600,k=512,n=2048）：

| 路径 | rel_rms |
|---|---:|
| fp16 反量化 + MLX fp16 matmul | **4.6e-4**（← 其实内部是 fp32 累加，误差全来自 fp16 输出舍入） |
| fp16 存 / fp32 累加 | 4.9e-8 |
| fp32 反量化 | 4.0e-7 |

对照：int8 激活量化本身的相对误差 ~1/254 = **4e-3**。也就是说 fp16 路线的误差比模型本来就要吃的
量化误差还小一个数量级 —— 这是 CER 只有 1.2% 的底气。

### 3.5 优化历程（同一台机器，bv58）

| 阶段 | 时间 | 倍实时 | 做了什么 |
|---|---:|---:|---|
| 朴素逐窗串行（首版） | 978 ms（含 700ms 首次编译） | 60x | 端口跑通 |
| 稳定态串行 | 254 ms | 231x | — |
| + 批窗口 B=4 | 191 ms | 306x | 窗口间无状态依赖 → 堆 batch，GEMM (390,512)→(1560,512) |
| + `mx.compile` 整窗 | 135 ms | 434x | 40 层图编译，融合 LN/GELU/add，消 dispatch |
| CLI 端到端（Rust + MLX，含 fbank） | **141 ms** | **416x** | 实测口径（Python 原型是 170ms/345x） |

**批窗口为什么有效**：引擎的窗口是 t1≈390 帧（blk=1600），单窗时 GEMM 太小，fp16 利用率只有 ~11%；
堆成 B=4 后到 ~45%。再往上（B=8）就饱和了，而且模型是 CTC 编码器、窗口之间确实没有状态依赖，
所以批窗口不改变数值结果（只改变窗口网格，尾部按真实长度收尾以免末字被截）。
**`mx.compile` 为什么有效**：40 层 × ~12 个算子 = ~480 次 dispatch，Python/Metal 编码开销在 t1=390 时
占比可观；编译后融合逐元素链，-29%。

### 3.6 尝试过但没用的

| 尝试 | 结果 |
|---|---|
| 加大窗口到 t1=1183（整文件一次解） | 更快（299x）但**文本退化**（汤屋→汤污、抗震→抗奋）：超出模型训练时的窗口长度 |
| `quantized_matmul`（int8 权重） | 慢 11~24%：Apple 没有 int8 算力，反量化纯属额外开销 |
| 单纯 `mx.compile` 不开批窗口 | 232x，只比串行好一点（瓶颈在 GEMM 太小） |
| 只融 q/k/v 成一个 GEMM | ±0（MLX 的 dispatch 不是主要瓶颈，`mx.compile` 才是） |

## 四、移植后的代码（在 Rust 引擎里，不是另起一套）

后端与 CUDA 版同构，`--cpu` / `--gpu` / `--mlx` 在同一个二进制里：

```
csrc_mlx/wasr_mlx.cc   MLX 计算层（对应 csrc/*.cu）：C ABI + 权重反量化 + 卷积前端 + 40 层 + argmax
src/mlx.rs             Rust FFI（对应 src/gpu.rs）：WlPlan/WlLayerPlan（repr(C)）+ 重量上传 + 单窗/批窗前向
src/model.rs           Model 多一个 mlx 字段，forward() 按 feature 分派；与 CUDA 共用同一份 blob/plan
src/stream.rs          finish_mlx_batched()：固定网格 + 批窗口 + 每窗取 [s, s+blk)
src/bin/wasr.rs        --mlx / --batch N（输出格式与 CUDA/CPU 后端完全一致）
build.rs               按 feature 编译：cuda → nvcc；mlx → c++ + libwasr_mlx.a + libmlx.dylib
```

```bash
python3 -m pip install --user mlx      # 提供 C++ 头文件与 libmlx.dylib，build.rs 自动定位
cargo build --release --no-default-features --features mlx
./target/release/wasr ./models/wtmodel --mlx --batch 4 a.wav b.wav
```

验证方式：从 hermes 取官方 `.xnet` 权重与测试 wav → M2 Pro 上跑 Rust+MLX → 与 hermes 上 CUDA 版的转录
逐字对比（bv58 CER 1.2%，real1 逐字一致）。

踩到的两个坑（已修，写在这里省得下次再踩）：

1. **C/Rust 结构体字段顺序必须逐字段一致**。一开始把 `q,k,v,o` 四个 offset 和四个 scale 分开排，
   偏移全错 → 段错误；现在 `wl_abi_sizes()` 会在启动时校验两边 `sizeof`，不一致直接报错。
2. **`vdotq_s32`（aarch64 SDOT）在 stable Rust 上还是 unstable**（`stdarch_neon_dotprod`），
   这就是"Rust 版在 Mac 上编不过"的根因之一。现在 build.rs 自动检测 nightly：nightly 走 SDOT，
   stable 退回标量（CPU 17.7x 实时，够用；MLX 后端不受影响）。

## 五、如果换 M5 机器会怎样

**硬件层面第一次不一样了**：M5 起 Metal 4 有 TensorOps（`MetalPerformancePrimitives` 的 `matmul2d`，
16×32×16 tile），原生支持 **INT8×INT8→INT32** / FP16×FP16→FP32 / BF16×BF16→FP32。
第三方在 M5 Pro（20 核 GPU）上实测手写 int8 GEMM：形状 (128,4096,4096) 0.361ms、≈**22 TOPS**，
(128,3584,18944) 0.794ms、≈22 TOPS，且逐位对拍 NumPy int64 通过。

**但现成代码直接上 M5，只能吃到 fp16 那一档**：

1. **MLX 用不上 TensorOps**。MLX 的 GEMM（`nax.h`）对量化权重是"反量化成 fp16/bf16 再算"，
   从来不走 int8×int8→int32；官方那个 int8 路径的 PR 默认关闭。所以现在这版 `csrc_mlx/` 后端在 M5 上
   仍按 fp16 ALU 跑。
2. `mx.fast.metal_kernel`（内联 MSL）也帮不上：它自动生成的 wrapper 会插入非一致控制流，
   与 `cooperative_tensor` 要求的"整个 simdgroup 均匀参与"冲突。要写 TensorOps kernel 得走
   MLX 的 C++ primitive API（`mx::Primitive` + `PipelineCache`），或者干脆跳出 MLX 自己发 command buffer。

**预期数字**（未实测，按核数与代际估算）：

| 机器 | 跑现在这份代码（fp16） | 再写 int8 TensorOps GEMM |
|---|---:|---:|
| M2 Pro 16 核（已实测） | 416x | 不适用（无 int8 硬件） |
| M5（10 核 GPU） | 估计 350~500x | 估计 600~900x |
| M5 Pro（16~20 核 GPU） | 估计 500~800x | 估计 900~1500x（接近 5060 Ti 的 1100~1300x） |

注意后一列是"另写一个 Metal 后端"的收益，不是改几个参数；而且我们的窗口形状 K=512 偏小，
TensorOps 的 16×32×16 tile 利用率不如 LLM prefill 那种 K=4096 的大矩阵，效率要打折扣。

**建议**：M5 到手先零改动跑一遍现在的 `--mlx` 后端（顺便重新调 `--blk/--batch`，M5 上最优值可能变），
拿到基线再决定值不值得写 int8 Metal kernel。反过来说，**M5 也是 Mac 上第一次有机会保住"逐字一致"
契约的硬件**（int8 整数累加与顺序无关），如果这个验收标准重要，M5 + 手写 Metal 是唯一可行组合。

## 六、如果要在 Mac 上继续做，按性价比排序

1. **M5+ 的 Metal 4 TensorOps**（`matmul2d` 原生 int8×int8→int32）：这代硬件才有 int8 矩阵单元，
   届时可以 1:1 平移 CUDA 版的 W8A8 契约，速度也有机会翻倍。M1–M4 无解。
2. ~~让 Rust 版在 macOS 上编得过~~ **已做**：CUDA 侧全部 feature gate、`build.rs` 按 feature 走，
   `cargo build --release --no-default-features --features mlx` 在 Apple Silicon 上直接能编出来。
3. **流式中间结果**：`Stream::partial()` 已有，CLI 没接；批模式不需要，流式场景要补。
4. **不要做**：CoreML/ANE 整体迁移（动态 per-row 量化 + 自定义 epilogue + 流式循环都不匹配，
   且公开先例里 ANE 普遍比 GPU 慢）；手写 Metal int8（除非要逐字一致，且性能不会比 MLX 高，
   因为 Apple GPU 的 int8 与 fp16 走同一套 ALU，只省一半权重带宽）。

## 七、复现

```bash
# Mac 上（Apple Silicon）
python3 -m pip install --user mlx          # 提供 MLX 的 C++ 头文件与 libmlx.dylib
cargo build --release --no-default-features --features mlx
./target/release/wasr ./models/wtmodel --mlx --batch 4 test/bv58.wav
WASR_MLX_DEBUG=1 ./target/release/wasr ... # 打点（定位崩溃/慢在哪一步）

# 基准脚本：scripts/mlx/bench_cpu_neon.rs（CPU int8 微基准，需要 nightly，
# stable 上 vdotq_s32 还是 unstable）
```

> 注：`scripts/mlx/bench_cpu_neon.rs` 需要用 nightly 编（`rustc +nightly -O`），
> 因为 stable 上 `vdotq_s32` 还是 unstable。
