//! 构建：按 feature 选后端
//!
//!   cuda（默认）→ nvcc 把 `csrc/*.cu` 编成单个 `libwasr_gpu.so`（符号前缀 wp_*/wk_*）
//!   mlx          → c++ 把 `csrc_mlx/wasr_mlx.cc` 编成 `libwasr_mlx.a`，并链接 MLX 的 `libmlx.dylib`
//!
//! macOS 上编译 MLX 后端：
//!   cargo build --release --no-default-features --features mlx
//! MLX 的 C++ 头文件和 libmlx.dylib 可从 `pip install mlx` 的 site-packages 里直接拿到
//! （<site-packages>/mlx/include + <site-packages>/mlx/lib），也可用 MLX_PREFIX / MLX_INCLUDE +
//! MLX_LIB_DIR 指定自建的 mlx-c 安装。找不到时本脚本会自动跑 python3 探测。
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(neon_dotprod)");
    // aarch64 的 SDOT 内建（vdotq_s32）在 stable 上还是 unstable（stdarch_neon_dotprod），
    // 所以只在 nightly 工具链上启用；stable 自动退回标量实现（CPU 慢，但不影响 MLX 后端）。
    if env::var("CARGO_CFG_TARGET_ARCH").ok().as_deref() == Some("aarch64") {
        let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
        if let Ok(o) = Command::new(&rustc).arg("--version").output() {
            if String::from_utf8_lossy(&o.stdout).contains("nightly") {
                println!("cargo:rustc-cfg=neon_dotprod");
            }
        }
    }
    if env::var("CARGO_FEATURE_MLX").is_ok() {
        build_mlx(&manifest, &out);
    }
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda(&manifest, &out);
    }
}

/// MLX 后端（Apple Silicon）
fn build_mlx(manifest: &Path, out: &Path) {
    let csrc = manifest.join("csrc_mlx");
    println!("cargo:rerun-if-changed=csrc_mlx/wasr_mlx.cc");
    for k in ["MLX_INCLUDE", "MLX_LIB_DIR", "MLX_PREFIX", "CXX"] {
        println!("cargo:rerun-if-env-changed={k}");
    }
    if env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("macos") {
        panic!("mlx feature 只支持 macOS（Apple Silicon）");
    }
    let (inc, lib) = mlx_paths();
    let cxx = env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let obj = out.join("wasr_mlx.o");
    let lib_a = out.join("libwasr_mlx.a");

    let st = Command::new(&cxx)
        .args(["-std=c++17", "-O2", "-fPIC", "-c"])
        .arg("-I")
        .arg(&inc)
        .arg(csrc.join("wasr_mlx.cc"))
        .arg("-o")
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("无法执行 {cxx}: {e}"));
    if !st.success() {
        panic!("MLX 后端编译失败（CXX={cxx}, include={}）", inc.display());
    }
    let st = Command::new("ar").arg("rcs").arg(&lib_a).arg(&obj).status().unwrap_or_else(|e| panic!("无法执行 ar: {e}"));
    if !st.success() {
        panic!("ar 打包 libwasr_mlx.a 失败");
    }

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=wasr_mlx");
    println!("cargo:rustc-link-lib=dylib=c++");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=mlx");
    // 运行时找 libmlx.dylib（它自己会从同目录找 mlx.metallib）
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
}

/// 依次尝试：环境变量 → 各 python 解释器的 site-packages 里的 mlx → homebrew /usr/local
fn mlx_paths() -> (PathBuf, PathBuf) {
    if let (Ok(i), Ok(l)) = (env::var("MLX_INCLUDE"), env::var("MLX_LIB_DIR")) {
        return (PathBuf::from(i), PathBuf::from(l));
    }
    if let Ok(p) = env::var("MLX_PREFIX") {
        let p = PathBuf::from(p);
        return (p.join("include"), p.join("lib"));
    }
    // 注意：cargo 的非交互环境里 `python3` 可能不是装了 mlx 的那个解释器，
    // 所以这里遍历常见解释器 + 常见 site-packages 位置。
    let mut pythons = vec!["python3".to_string()];
    for p in ["/usr/bin/python3", "/opt/homebrew/bin/python3", "/usr/local/bin/python3"] {
        pythons.push(p.to_string());
    }
    if let Ok(home) = env::var("HOME") {
        // 覆盖 `pip install --user` 装到 ~/Library/Python/<ver>/lib/python/site-packages 的情况
        if let Ok(rd) = std::fs::read_dir(Path::new(&home).join("Library/Python")) {
            for e in rd.flatten() {
                pythons.push(format!(
                    "{}/Library/Python/{}/lib/python/site-packages",
                    home,
                    e.file_name().to_string_lossy()
                ));
            }
        }
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    for py in &pythons {
        let (cmd, arg): (String, Vec<String>) = if py.ends_with("site-packages") {
            ("".to_string(), vec![py.clone()])
        } else {
            (
                py.clone(),
                vec!["-c".to_string(), "import sysconfig;print(sysconfig.get_paths()['purelib'])".to_string()],
            )
        };
        if cmd.is_empty() {
            candidates.push(PathBuf::from(&arg[0]));
            continue;
        }
        if let Ok(o) = Command::new(&cmd).args(&arg).output() {
            if o.status.success() {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if !s.is_empty() {
                    candidates.push(PathBuf::from(s));
                }
            }
        }
    }
    // homebrew 的 python 也会把 mlx 放进 lib/python3.x/site-packages
    for root in ["/opt/homebrew/lib", "/usr/local/lib"] {
        if let Ok(rd) = std::fs::read_dir(root) {
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().to_string();
                if let Some(rest) = n.strip_prefix("python3") {
                    if !rest.is_empty() {
                        candidates.push(Path::new(root).join(&n).join("site-packages"));
                    }
                }
            }
        }
    }
    for p in ["/opt/homebrew", "/usr/local"] {
        let p = PathBuf::from(p);
        if p.join("include/mlx/mlx.h").exists() && p.join("lib/libmlx.dylib").exists() {
            return (p.join("include"), p.join("lib"));
        }
    }
    for site in candidates {
        let p = site.join("mlx");
        if p.join("include/mlx/mlx.h").exists() && p.join("lib/libmlx.dylib").exists() {
            return (p.join("include"), p.join("lib"));
        }
    }
    panic!(
        "找不到 MLX 的 C++ 头/库。办法（任选其一）：\n\
         \x20 1) python3 -m pip install --user mlx   （本脚本会去各 python 的 site-packages 里找 mlx/include 与 mlx/lib）\n\
         \x20 2) MLX_PREFIX=/path/to/mlx（其下有 include/mlx 与 lib/libmlx.dylib）\n\
         \x20 3) MLX_INCLUDE=... MLX_LIB_DIR=... 分开指定"
    );
}

/// CUDA 后端（NVIDIA GPU）
fn build_cuda(manifest: &Path, out: &Path) {
    let csrc = manifest.join("csrc");
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
