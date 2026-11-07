//! Apple Silicon / MLX 后端（`--mlx`）。对应 CUDA 版的 `src/gpu.rs`。
//!
//! 分工与 CUDA 路径完全一致：Rust 负责 fbank / .xnet 解析 / 权重打包 / 分块解码 / CTC / CLI，
//! `csrc_mlx/wasr_mlx.cc` 只负责"一个窗口的前向"：[t][40] 特征帧 → t1 个 token id。
//!
//! 为什么不逐位复刻 int8：Apple GPU（M1–M4）没有 int8 矩阵单元，MLX 的 matmul 也不收整数。
//! 权重在 create 时一次性反量化成 fp16，前向全 fp16（matmul 内部 fp32 累加），
//! 实测相对误差 4.6e-4，比原版 int8 激活量化本身的 4e-3 还小一个数量级。
#![allow(clippy::too_many_arguments)]

use crate::fbank::DIM;
use std::ffi::c_void;

/// 一层的权重位置（blob 内字节偏移）+ per-tensor 量化 scale。
/// 字段顺序必须与 csrc_mlx/wasr_mlx.cc 的 wl_layer_plan 完全一致。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct WlLayerPlan {
    pub pn_g: usize,
    pub pn_b: usize,
    pub mn_g: usize,
    pub mn_b: usize,
    pub q: usize,
    pub qs: f32,
    pub k: usize,
    pub ks: f32,
    pub v: usize,
    pub vs: f32,
    pub o: usize,
    pub os: f32,
    pub f1: usize,
    pub f1s: f32,
    pub f1b: usize,
    pub f2: usize,
    pub f2s: f32,
    pub f2b: usize,
}

/// 整个模型的权重位置表（对应 wl_plan）。
#[repr(C)]
pub struct WlPlan {
    pub layers: *const WlLayerPlan,
    pub n_layers: i32,
    pub lin: usize,
    pub lin_s: f32,
    pub lin_b: usize,
    pub out: usize,
    pub out_s: f32,
    pub out_b: usize,
    pub vocab: i32,
    pub d: i32,
    pub ff: i32,
    pub flat: i32,
    pub heads: i32,
    pub c1: usize,
    pub c2: usize,
    pub c3: usize,
}

extern "C" {
    fn wl_create(blob: *const c_void, bytes: usize, plan: *const WlPlan) -> *mut c_void;
    fn wl_destroy(c: *mut c_void);
    fn wl_forward(c: *mut c_void, feats: *const f32, t: i32, ids: *mut u32) -> i32;
    fn wl_forward_batch(c: *mut c_void, feats: *const f32, b: i32, t: i32, ids: *mut u32) -> i32;
    fn wl_abi_sizes(layer_size: *mut i32, plan_size: *mut i32);
}

/// 启动时校验 C/Rust 两边结构体大小一致（字段顺序错了会直接段错误，不如在这里报清楚）
fn abi_check() {
    let (mut l, mut p) = (0i32, 0i32);
    unsafe { wl_abi_sizes(&mut l, &mut p) };
    assert_eq!(
        l as usize,
        std::mem::size_of::<WlLayerPlan>(),
        "wl_layer_plan ABI 不一致：C={l} Rust={}",
        std::mem::size_of::<WlLayerPlan>()
    );
    assert_eq!(
        p as usize,
        std::mem::size_of::<WlPlan>(),
        "wl_plan ABI 不一致：C={p} Rust={}",
        std::mem::size_of::<WlPlan>()
    );
}

pub struct Mlx {
    ctx: *mut c_void,
    pub vocab: usize,
}

unsafe impl Send for Mlx {}
unsafe impl Sync for Mlx {}

/// 输入 t 个特征帧时输出的 token 数（卷积时间 stride 5）
#[inline]
pub fn t1_of(t: usize) -> usize {
    if t < 5 {
        0
    } else {
        (t - 5) / 5 + 1
    }
}

fn as_f32(x: &[[f32; DIM]]) -> &[f32] {
    // [[f32; DIM]] 与 [f32] 布局相同
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len() * DIM) }
}

impl Mlx {
    pub fn new(blob: &[u8], plan: &WlPlan) -> Mlx {
        abi_check();
        let ctx = unsafe { wl_create(blob.as_ptr() as *const c_void, blob.len(), plan) };
        assert!(!ctx.is_null(), "wl_create 失败（MLX 权重上传）");
        Mlx { ctx, vocab: plan.vocab as usize }
    }

    /// 单窗前向：[t][40] f32 特征 → t1 个 token id
    pub fn forward(&self, feats: &[[f32; DIM]]) -> Vec<u32> {
        let t = feats.len();
        let t1 = t1_of(t);
        if t1 == 0 {
            return Vec::new();
        }
        let mut ids = vec![0u32; t1];
        let rc = unsafe { wl_forward(self.ctx, as_f32(feats).as_ptr(), t as i32, ids.as_mut_ptr()) };
        assert_eq!(rc as usize, t1, "wl_forward rc={rc} (t={t})");
        ids
    }

    /// 批窗前向：B 个等长窗口一次算完（B 个窗口的 GEMM 合成一个大 GEMM）
    pub fn forward_batch(&self, windows: &[Vec<[f32; DIM]>], t: usize) -> Vec<Vec<u32>> {
        let b = windows.len();
        let t1 = t1_of(t);
        if b == 0 || t1 == 0 {
            return Vec::new();
        }
        let mut flat = vec![0f32; b * t * DIM];
        for (k, w) in windows.iter().enumerate() {
            assert_eq!(w.len(), t, "批窗口长度必须一致");
            flat[k * t * DIM..(k + 1) * t * DIM].copy_from_slice(as_f32(w));
        }
        let mut ids = vec![0u32; b * t1];
        let rc = unsafe {
            wl_forward_batch(self.ctx, flat.as_ptr(), b as i32, t as i32, ids.as_mut_ptr())
        };
        assert_eq!(rc as usize, t1, "wl_forward_batch rc={rc} (b={b} t={t})");
        ids.chunks(t1).map(|c| c.to_vec()).collect()
    }
}

impl Drop for Mlx {
    fn drop(&mut self) {
        unsafe { wl_destroy(self.ctx) }
    }
}
