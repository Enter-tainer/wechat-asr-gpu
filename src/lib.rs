//! wtlasr —— 微信输入法离线 ASR 引擎（上游 `voicekey/crates/wtlocal`）的纯 Rust + CUDA 复刻。
//!
//! 结构：Rust 负责"引擎外壳"（fbank 特征、.xnet 容器解析、权重打包、分块解码、CTC 折叠、CLI），
//! 计算在 CUDA 里（`csrc/`，编成单个 `libwtlasr_gpu.so`）。CPU 后端保留，用于无 GPU 机器与对照。
//!
//! GPU 路径：`WTL_GPU=1`（或 CLI 的 `--gpu`）。
mod fbank;
mod gpu;
mod kernels;
mod model;
pub mod pack;
mod stream;
mod xnet;

pub use model::Model;
pub use stream::Stream;

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

static CACHE: Mutex<Option<(PathBuf, Arc<Model>)>> = Mutex::new(None);

/// 加载模型（同一目录只加载一次；GPU 后端会在首次加载时把 130MB 权重上传到显存）
pub fn load(dir: &Path) -> Result<Arc<Model>> {
    let mut c = CACHE.lock().unwrap();
    if let Some((d, m)) = c.as_ref() {
        if d == dir {
            return Ok(m.clone());
        }
    }
    let m = Arc::new(Model::load(dir)?);
    *c = Some((dir.to_path_buf(), m.clone()));
    Ok(m)
}
