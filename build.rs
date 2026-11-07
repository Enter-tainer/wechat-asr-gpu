//! 构建：把 csrc/ 下的 CUDA 源码编成**单个** `libwasr_gpu.so`，然后链接进 Rust。
//!
//! 三个 .cu 的符号前缀互不冲突，合并成一个动态库：
//!   wtlgpu.cu          -> wp_*  int8 GEMM（cuBLASLt）+ CUDA graph 封装
//!   wtl_kern.cu        -> wk_*  卷积前端 / norm+quant / gelu+quant / argmax
//!   wk_attention_fast.cu -> wk_attention_cublas / wk_attention_tiled
//!   wk_attn_api.cu     -> wk_attention_fast（运行时分派）
//!
//! 可用环境变量覆盖：
//!   NVCC            nvcc 路径（默认 /usr/local/cuda/bin/nvcc，找不到就用 PATH 里的 nvcc）
//!   WTL_CUDA_ARCH   -arch 值（默认 native；老工具链可设 sm_90 / sm_120 等）
//!   WTL_SKIP_CUDA=1 跳过 CUDA 编译（仅调试 Rust 侧语法，运行时 GPU 不可用）
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let csrc = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR")).join("csrc");

    let sources = ["wtlgpu.cu", "wtl_kern.cu", "wk_attention_cublas.cu", "wk_attention_fast.cu", "wk_attn_api.cu"];
    for s in sources {
        println!("cargo:rerun-if-changed=csrc/{s}");
    }
    println!("cargo:rerun-if-changed=csrc/wtlgpu.h");

    if env::var("WTL_SKIP_CUDA").ok().as_deref() == Some("1") {
        println!("cargo:warning=WTL_SKIP_CUDA=1：跳过 CUDA 编译，GPU 后端不可用");
        return;
    }

    let nvcc = env::var("NVCC").unwrap_or_else(|_| {
        let p = PathBuf::from("/usr/local/cuda/bin/nvcc");
        if p.exists() {
            p.to_string_lossy().to_string()
        } else {
            "nvcc".to_string()
        }
    });
    let arch = env::var("WTL_CUDA_ARCH").unwrap_or_else(|_| "native".to_string());
    let so = out.join("libwasr_gpu.so");

    let mut cmd = Command::new(&nvcc);
    cmd.arg("-O3")
        .arg(format!("-arch={arch}"))
        .arg("-shared")
        .arg("--compiler-options")
        .arg("-fPIC")
        .arg("-Xcompiler")
        .arg("-Wno-unknown-pragmas")
        .arg("-o")
        .arg(&so)
        .arg("-I")
        .arg(&csrc)
        .arg("-L/usr/local/cuda/lib64")
        .arg("-lcublas")
        .arg("-lcublasLt")
        .arg("-lcudart")
        .arg("-Xlinker")
        .arg("-rpath")
        .arg("-Xlinker")
        .arg("/usr/local/cuda/lib64");
    for s in sources {
        cmd.arg(csrc.join(s));
    }

    let status = cmd.status().unwrap_or_else(|e| panic!("无法执行 nvcc ({nvcc}): {e}；可用 NVCC=... 覆盖"));
    if !status.success() {
        panic!("CUDA 编译失败（nvcc {nvcc}, arch {arch}）。可设 NVCC / WTL_CUDA_ARCH 覆盖，或 WTL_SKIP_CUDA=1 跳过");
    }

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib=wasr_gpu");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out.display());
}
