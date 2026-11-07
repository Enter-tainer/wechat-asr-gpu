//! wtlasr —— 微信输入法离线 ASR 引擎（上游 `voicekey/crates/wtlocal`）的纯 Rust 复刻，
//! 三套计算后端：CUDA（默认 feature）/ Apple Silicon MLX（`--features mlx`）/ CPU。
//!
//! aarch64 的 SDOT 内建（vdotq_s32）在 stable Rust 上还是 unstable，所以只在 nightly 上启用
//! （build.rs 自动检测并设置 `neon_dotprod` cfg）；stable 自动退回标量实现。
#![cfg_attr(neon_dotprod, feature(stdarch_neon_dotprod))]
//!
//! 结构：Rust 负责"引擎外壳"（fbank 特征、.xnet 容器解析、权重打包、分块解码、CTC 折叠、CLI），
//! 计算在 CUDA 里（`csrc/`，编成单个 `libwasr_gpu.so`）或 MLX 里（`csrc_mlx/` + `src/mlx.rs`）。
//! CPU 后端保留，用于无 GPU 机器与对照。
//!
//! GPU 路径：`WTL_GPU=1`（或 CLI 的 `--gpu`）；MLX 路径：`WTL_MLX=1`（或 `--mlx`）。
mod fbank;
#[cfg(feature = "cuda")]
mod gpu;
#[cfg(feature = "mlx")]
mod mlx;
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

/// 释放模型（触发出 GPU 上下文销毁：写回 cublasLt 算法缓存、释放显存）。
/// 静态缓存会一直持有 Arc，所以进程结束前要显式调用一次。
pub fn unload() {
    *CACHE.lock().unwrap() = None;
}

/// 整段转写（批模式入口）：MLX 后端且 `batch > 1` 时走批窗口，其余情况走原来的串行分块。
/// CUDA / CPU 后端不认 `batch`，行为与以前完全一致。
pub fn transcribe(m: Arc<Model>, pcm: &[i16], batch: usize) -> String {
    #[cfg(feature = "mlx")]
    if m.mlx_enabled() && batch > 1 {
        let mut s = Stream::new(m);
        s.push_raw(pcm);
        return s.finish_mlx_batched(batch);
    }
    let _ = batch;
    let mut s = Stream::new(m);
    s.push(pcm);
    s.finish()
}