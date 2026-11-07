"""Apple Silicon / MLX 后端的微信离线 ASR 引擎（Rust + CUDA 版的 MLX 移植）。

入口：
    python -m mlx_backend.cli <模型目录> a.wav b.wav ...

模块：
    xnet.py     .xnet 容器解析（对应 src/xnet.rs）
    fbank.py    40 维 fbank 特征（对应 src/fbank.rs）
    weights.py  模型目录 → MLX 张量（对应 src/model.rs 的 load/Blob）
    model.py    卷积前端 + 40 层 Transformer + CTC（对应 src/model.rs 的 forward）
    decode.py   分块解码 + CTC 折叠（对应 src/stream.rs）
    cli.py      批模式 CLI（对应 src/bin/wasr.rs）
"""
from .decode import Cfg, Stream
from .model import Model

__all__ = ["Model", "Stream", "Cfg"]
