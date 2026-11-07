# 优化记录（已落地 / 已回退 / 剩余）

README 只放结论，这里是取舍和实测依据。CUDA 侧的完整报告见 [REPORT.md](REPORT.md)，
Mac 侧见 [MLX-REPORT.md](MLX-REPORT.md)。

## CUDA（NVIDIA）

| 优化 | 结果 | 说明 |
|---|---|---|
| 整窗 CUDA graph | ✅ | 把 linear + 40 层 + 输出层 + argmax 约 806 个 kernel 捕获成一张图，同 t1 的窗口直接 replay（host 入队 3ms/窗口 → ~1.5µs）；long593 -6%，输出逐字一致。只对总特征帧 ≥20000（200s）的音频启用，短音频捕获开销 > 省下的时间。`WTL_GRAPH=1/0` 强制开/关 |
| fbank 并行化 | ✅ | 批模式整段喂入时按帧并行算 FFT/mel；滚动均值仍按原顺序累加，与单线程逐位一致。58.65s 音频的 fbank 29ms → ~4ms |
| cuBLASLt 算法选择缓存 | ✅ | 每形状几十 ms 的候选实测结果持久化到 `~/.cache/wasr/lt_algo.bin`（记录 GPU 名 / driver / CUDA 版本做失效保护）。单条 bv58 143ms → 120ms |
| conv2/conv3 shared-memory tiling | ❌ 回退 | bv58 121→131ms、long593 555→654ms。卷积是 latency-bound 不是吞吐-bound，shared 用量把 occupancy 压到 ~2 block/SM，反而不如"每输出一线程" |
| 剩余候选 | — | ①norm/quant 并行归约（4 个融合算子占窗口 13%，均值/方差现在 thread0 串行累加，≈-6%）②attention 换 fp16 / TileLang flash-attn（cuBLAS 版 18.8µs/次，TileLang 实测 7.15µs，C ABI 未验收）③fbank GPU 化（并行化后只剩 2~6%） |

单窗口（t=150）拆解约 7ms：attention 0.75、conv2+conv3 1.3、norm/quant 类 1.3、int8 GEMM GPU 0.9、
GEMM host 入队 0.0015（graph 之后）、epilogue+gelu 1.0（单位 ms）。

## Apple Silicon（MLX）

| 优化 | 结果 | 说明 |
|---|---|---|
| 批窗口 | ✅ | 这是 CTC 编码器，窗口之间没有状态依赖（上下文只来自音频 prev/post 重叠），`--batch N` 把 N 个固定网格窗口堆成一个 batch：GEMM 从 (390,512) 变成 (N·390,512)，fp16 利用率 11% → 45%（bv58 254ms → 191ms） |
| `mx.compile` 整窗图 | ✅ | 40 层约 480 次 dispatch 融合成一张图，按 (B,t1) 缓存，-29%（191ms → 141ms）。注意 t1 必须显式传入，不能从 t 反推（`(t1-1)*5+1 ≠ t`，会丢余数） |
| q/k/v 融合 + 用 `fast::scaled_dot_product_attention` | ✅ | attention 占窗口 14%，SDPA 在 t=390 也有 4.26 TFLOPS |
| 加大窗口到 t1≈1183（整文件一次解） | ❌ 回退 | 更快（299x）但**文本退化**（汤屋→汤污、抗震→抗奋）：超出模型训练时的窗口长度 |
| `quantized_matmul`（int8 权重） | ❌ | 比 fp16 GEMM 慢 11~24%：Apple 没有 int8 算力，反量化纯属额外开销 |
| M5+ 的 Metal 4 TensorOps | 待硬件 | M5 起才有原生 int8 `matmul2d`，届时可以 1:1 平移 CUDA 版的 W8A8 契约（也是 Mac 上唯一能"逐字一致"的路），见 MLX-REPORT.md 第五节 |
