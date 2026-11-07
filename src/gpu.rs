//! GPU 后端（WTL_GPU=1 时启用）。
//!
//! 分工：
//!   * 矩阵乘 / 激活量化融合 → libwtlgpu.so（cublasLt int8，gpu-probe 交付，与 CPU int32 逐元素一致）
//!   * 卷积前端 / attention / norm+quant / gelu+quant / argmax → libwtlkern.so（见 wtlkern.cu）
//!
//! 全部算子都跑在 **legacy stream 0** 上：不同 kernel 之间天然全序，不需要 event/sync。
//! 一次 forward 只在取回 token id 前 `wp_sync` 一次。
#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;

#[link(name = "wasr_gpu")]
extern "C" {
    fn wp_create(w_host: *const c_void, bytes: usize) -> *mut c_void;
    fn wp_destroy(c: *mut c_void);
    fn wp_alloc(bytes: usize) -> *mut c_void;
    fn wp_free(p: *mut c_void);
    fn wp_h2d(dev: *mut c_void, host: *const c_void, bytes: usize) -> i32;
    fn wp_d2h(host: *mut c_void, dev: *const c_void, bytes: usize) -> i32;
    fn wp_devptr(c: *mut c_void, off: usize) -> *mut c_void;
    fn wp_sync(c: *mut c_void) -> i32;
    fn wp_gemm(
        c: *mut c_void,
        xq: *const i8,
        xs: *const f32,
        t: i32,
        k: i32,
        w_off: usize,
        n: i32,
        ws: f32,
        bias_off: usize,
        out: *mut f32,
    ) -> i32;
    fn wp_stream(c: *mut c_void) -> *mut c_void;
}

extern "C" {
    fn wk_attention_fast(
        s: *mut c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        t: i32,
        d: i32,
        heads: i32,
        scratch: *mut f32,
        out: *mut f32,
    ) -> i32;
}
extern "C" {
    fn wk_front(
        s: *mut c_void,
        x: *const f32,
        c1: *const f32,
        c2: *const f32,
        c3: *const f32,
        t: i32,
        t1: i32,
        y1: *mut f32,
        y2: *mut f32,
        y3: *mut f32,
    );
    fn wk_norm_quant(s: *mut c_void, h: *const f32, g: *const f32, b: *const f32, t: i32, d: i32, xq: *mut i8, xs: *mut f32);
    fn wk_add_norm_quant(
        s: *mut c_void,
        h: *mut f32,
        o: *const f32,
        g: *const f32,
        b: *const f32,
        t: i32,
        d: i32,
        xq: *mut i8,
        xs: *mut f32,
    );
    fn wk_quant(s: *mut c_void, x: *const f32, t: i32, k: i32, xq: *mut i8, xs: *mut f32);
    fn wk_gelu_quant(s: *mut c_void, x: *mut f32, t: i32, k: i32, xq: *mut i8, xs: *mut f32);

    fn wk_add(s: *mut c_void, h: *mut f32, a: *const f32, t: i32, d: i32);
    fn wk_argmax(s: *mut c_void, x: *const f32, t: i32, n: i32, out: *mut u32);
}

pub const NO_BIAS: usize = usize::MAX;
pub const DIM: usize = 40;

fn alloc(bytes: usize) -> *mut c_void {
    let p = unsafe { wp_alloc(bytes.max(1)) };
    assert!(!p.is_null(), "wp_alloc({bytes}) 失败");
    p
}

pub struct Gpu {
    ctx: *mut c_void,
    /// libwtlgpu 的内部 stream：GEMM 与我的小算子**都**跑在这一条上，保证发起顺序即执行顺序
    stream: *mut c_void,
    pub tmax: usize,
    /// 一次 forward 允许的最大输入特征帧数（t1 = (t-5)/5+1 ≤ tmax）
    pub tmax_feat: usize,
    pub vocab: usize,
    pub x: *mut f32,
    pub y1: *mut f32,
    pub y2: *mut f32,
    pub y3: *mut f32,
    pub y3q: *mut i8,
    pub ys: *mut f32,
    pub h: *mut f32,
    pub xq: *mut i8,
    pub xs: *mut f32,
    pub q: *mut f32,
    pub k: *mut f32,
    pub v: *mut f32,
    pub att: *mut f32,
    pub scores: *mut f32,
    pub o: *mut f32,
    pub aq: *mut i8,
    pub asc: *mut f32,
    pub f: *mut f32,
    pub fq: *mut i8,
    pub fsc: *mut f32,
    pub lo: *mut f32,
    pub ids: *mut u32,
}

unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Gpu {
    pub fn new(blob: &[u8], tmax: usize, d: usize, ff: usize, vocab: usize, flat: usize, heads: usize) -> Gpu {
        let tmax_feat = 5 * (tmax - 1) + 5;
        unsafe {
            let ctx = wp_create(blob.as_ptr() as *const c_void, blob.len());
            assert!(!ctx.is_null(), "wp_create 失败（权重 blob 上传）");
            let stream = wp_stream(ctx);
            Gpu {
                ctx,
                stream,
                tmax,
                tmax_feat,
                vocab,
                x: alloc(tmax_feat * DIM * 4) as *mut f32,
                y1: alloc(tmax_feat * 32 * DIM * 4) as *mut f32,
                y2: alloc(tmax * 64 * 41 * 4) as *mut f32,
                y3: alloc(tmax * flat * 4) as *mut f32,
                y3q: alloc(tmax * flat) as *mut i8,
                ys: alloc(tmax * 4) as *mut f32,
                h: alloc(tmax * d * 4) as *mut f32,
                xq: alloc(tmax * d) as *mut i8,
                xs: alloc(tmax * 4) as *mut f32,
                q: alloc(tmax * d * 4) as *mut f32,
                k: alloc(tmax * d * 4) as *mut f32,
                v: alloc(tmax * d * 4) as *mut f32,
                att: alloc(tmax * d * 4) as *mut f32,
                scores: alloc(heads * tmax * tmax * 4) as *mut f32,
                o: alloc(tmax * d * 4) as *mut f32,
                aq: alloc(tmax * ff.max(d)) as *mut i8,
                asc: alloc(tmax * 4) as *mut f32,
                f: alloc(tmax * ff * 4) as *mut f32,
                fq: alloc(tmax * ff) as *mut i8,
                fsc: alloc(tmax * 4) as *mut f32,
                lo: alloc(tmax * vocab * 4) as *mut f32,
                ids: alloc(tmax * 4) as *mut u32,
            }
        }
    }

    /// blob 内偏移 → 显存指针（norm 的 gamma/beta、卷积权重、bias）
    pub fn dev(&self, off: usize) -> *mut c_void {
        unsafe { wp_devptr(self.ctx, off) }
    }

    pub fn sync(&self) {
        let rc = unsafe { wp_sync(self.ctx) };
        assert_eq!(rc, 0, "wp_sync 失败 rc={rc}");
    }

    /// 上传特征帧 x[t][40]
    pub fn upload_feats(&self, x: &[[f32; DIM]], t: usize) {
        let rc = unsafe { wp_h2d(self.x as *mut c_void, x.as_ptr() as *const c_void, t * DIM * 4) };
        assert_eq!(rc, 0, "上传特征帧失败");
    }

    pub fn upload_y3(&self, q: &[i8], s: &[f32]) {
        assert_eq!(unsafe { wp_h2d(self.y3q as *mut c_void, q.as_ptr() as *const c_void, q.len()) }, 0);
        assert_eq!(unsafe { wp_h2d(self.ys as *mut c_void, s.as_ptr() as *const c_void, s.len() * 4) }, 0);
    }

    pub fn download_ids(&self, t: usize) -> Vec<u32> {
        let mut out = vec![0u32; t];
        assert_eq!(unsafe { wp_d2h(out.as_mut_ptr() as *mut c_void, self.ids as *const c_void, t * 4) }, 0);
        out
    }

    /// GPU 卷积前端：x[t][40] → y3[t1][FLAT]
    pub fn front(&self, t: usize, t1: usize, c1: usize, c2: usize, c3: usize) {
        unsafe {
            wk_front(
                self.stream,
                self.x,
                self.dev(c1) as *const f32,
                self.dev(c2) as *const f32,
                self.dev(c3) as *const f32,
                t as i32,
                t1 as i32,
                self.y1,
                self.y2,
                self.y3,
            )
        }
    }

    /// 把 y3[t1][FLAT] f32 逐行量化为 int8（与 CPU quant() 同口径）
    pub fn quant_y3(&self, t1: usize, flat: usize) {
        unsafe { wk_quant(self.stream, self.y3, t1 as i32, flat as i32, self.y3q, self.ys) }
    }

    /// int8 GEMM：out[t][n] = (xq·wᵀ)·xs·ws + bias（bias 走权重 blob 偏移，内部流 = wp_stream）
    pub fn gemm(&self, xq: *const i8, xs: *const f32, t: usize, k: usize, w: usize, n: usize, ws: f32, bias: usize, out: *mut f32) {
        let bias_off = if bias == NO_BIAS { usize::MAX } else { bias };
        let rc = unsafe { wp_gemm(self.ctx, xq, xs, t as i32, k as i32, w, n as i32, ws, bias_off, out) };
        assert_eq!(rc, 0, "wp_gemm rc={rc} (t={t} k={k} n={n})");
    }
    /// h → layernorm(g,b) → int8
    pub fn norm_quant(&self, t: usize, d: usize, g: usize, b: usize, xq: *mut i8, xs: *mut f32) {
        unsafe {
            wk_norm_quant(self.stream, self.h, self.dev(g) as *const f32, self.dev(b) as *const f32, t as i32, d as i32, xq, xs)
        }
    }

    /// h += o，然后对 h 做 layernorm(g,b) → int8
    pub fn add_norm_quant(&self, t: usize, d: usize, g: usize, b: usize, xq: *mut i8, xs: *mut f32) {
        unsafe {
            wk_add_norm_quant(
                self.stream,
                self.h,
                self.o,
                self.dev(g) as *const f32,
                self.dev(b) as *const f32,
                t as i32,
                d as i32,
                xq,
                xs,
            )
        }
    }

    pub fn quant(&self, x: *const f32, t: usize, k: usize, xq: *mut i8, xs: *mut f32) {
        unsafe { wk_quant(self.stream, x, t as i32, k as i32, xq, xs) }
    }

    /// self.f 原地 gelu 后逐行量化
    pub fn gelu_quant(&self, t: usize, k: usize, xq: *mut i8, xs: *mut f32) {
        unsafe { wk_gelu_quant(self.stream, self.f, t as i32, k as i32, xq, xs) }
    }

    /// attention（libattn：cuBLAS batched sgemm + 连续内存 softmax，对数已与 CPU 参考对拍）
    pub fn attention(&self, t: usize, d: usize, heads: usize, q: *const f32, k: *const f32, v: *const f32, out: *mut f32) {
        let rc = unsafe { wk_attention_fast(self.stream, q, k, v, t as i32, d as i32, heads as i32, self.scores, out) };
        assert_eq!(rc, 0, "wk_attention_fast rc={rc} (t={t})");
    }

    pub fn add(&self, h: *mut f32, a: *const f32, t: usize, d: usize) {
        unsafe { wk_add(self.stream, h, a, t as i32, d as i32) }
    }

    pub fn argmax(&self, x: *const f32, t: usize, n: usize, out: *mut u32) {
        unsafe { wk_argmax(self.stream, x, t as i32, n as i32, out) }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            for p in [
                self.x as *mut c_void,
                self.y1 as *mut c_void,
                self.y2 as *mut c_void,
                self.y3 as *mut c_void,
                self.y3q as *mut c_void,
                self.ys as *mut c_void,
                self.h as *mut c_void,
                self.xq as *mut c_void,
                self.xs as *mut c_void,
                self.q as *mut c_void,
                self.k as *mut c_void,
                self.v as *mut c_void,
                self.att as *mut c_void,
                self.scores as *mut c_void,
                self.o as *mut c_void,
                self.aq as *mut c_void,
                self.asc as *mut c_void,
                self.f as *mut c_void,
                self.fq as *mut c_void,
                self.fsc as *mut c_void,
                self.lo as *mut c_void,
                self.ids as *mut c_void,
            ] {
                wp_free(p);
            }
            wp_destroy(self.ctx);
        }
    }
}
