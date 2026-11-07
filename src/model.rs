//! 声学模型：卷积前端 + 40 层 pre-norm Transformer + CTC 输出层；int8 权重直接 mmap 自 xnet
//!
//! 两条后端：
//!   * CPU（默认）：rayon 线程池 + 手写 AVX2/NEON int8 点积
//!   * GPU（`WTL_GPU=1`）：矩阵乘走 libwtlgpu.so 的 cublasLt int8，小算子走 libwtlkern.so，
//!     详见 src/gpu.rs。GPU 路径的 int8 结果与 CPU 逐元素相同，仅归一化/注意力刻意复刻
//!     CPU 的浮点求和顺序以保持可对拍。
use crate::fbank::DIM;
use crate::gpu::{Gpu, NO_BIAS};
use crate::kernels::{dotf, qgemm, quant, QAct};
use crate::xnet::{self, Rec, F32, I8};
use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

const D: usize = 512;
const HEADS: usize = 8;
const HD: usize = D / HEADS;
const FF: usize = 2048;
const LAYERS: usize = 40;
const F1: usize = 32; // 卷积通道
const F2: usize = 64;
const F3: usize = 192;
const K2: usize = F1 * 49;
const K3: usize = F2 * 9;
const W2: usize = DIM + 1; // 第二层卷积后频率维补一列 0
const FLAT: usize = F3 * 20;
/// GPU 路径一次能吃的最大输出帧数（t1）；超过则退回 CPU（正常流式窗口 t1 ≤ 400）
const TMAX: usize = 512;

struct Q {
    off: usize,
    n: usize,
    k: usize,
    s: f32,
}

struct Layer {
    pn: (Vec<f32>, Vec<f32>),
    mn: (Vec<f32>, Vec<f32>),
    q: Q,
    k: Q,
    v: Q,
    o: Q,
    f1: Q,
    f1b: Vec<f32>,
    f2: Q,
    f2b: Vec<f32>,
}

/// GPU 权重 blob 里一个 int8 权重矩阵的位置（256B 对齐的偏移）
#[derive(Clone, Copy)]
struct PlanQ {
    off: usize,
    s: f32,
}

struct PlanLayer {
    pn_g: usize,
    pn_b: usize,
    mn_g: usize,
    mn_b: usize,
    q: PlanQ,
    k: PlanQ,
    v: PlanQ,
    o: PlanQ,
    f1: PlanQ,
    f1b: usize,
    f2: PlanQ,
    f2b: usize,
}

struct Plan {
    layers: Vec<PlanLayer>,
    lin: PlanQ,
    lin_b: usize,
    out: PlanQ,
    out_b: usize,
    /// 输出层列数（= 词表长度向上补齐到 4 的倍数；cublasLt int8 要求 GEMM 的 M 维 %4==0）
    out_n: usize,
    /// 卷积前端三层权重（GPU 前端用）
    c1: usize,
    c2: usize,
    c3: usize,
}

pub struct Model {
    map: Mmap,
    c1: Vec<f32>,
    c2: Vec<f32>,
    c3: Vec<f32>,
    lin: Q,
    lin_b: Vec<f32>,
    layers: Vec<Layer>,
    out: Q,
    out_b: Vec<f32>,
    vocab: Vec<String>,
    gpu: Option<Gpu>,
    plan: Option<Plan>,
}

struct Loader<'a> {
    d: &'a [u8],
    r: HashMap<String, Rec>,
}

impl Loader<'_> {
    fn rec(&self, name: &str) -> Result<&Rec> {
        self.r.get(name).with_context(|| format!("模型缺少 {name}"))
    }
    fn f32(&self, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
        let r = self.rec(name)?;
        if r.dtype != F32 || r.dims != dims {
            bail!("{name} 形状不符：{:?}", r.dims);
        }
        Ok(self.d[r.off..r.off + r.len].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }
    fn q(&self, name: &str, n: usize, k: usize) -> Result<Q> {
        let name = format!("{name}.weight_quantized_s8");
        let r = self.rec(&name)?;
        if r.dtype != I8 || r.dims.len() < 2 || r.dims[..2] != [n, k] || r.len != n * k {
            bail!("{name} 形状不符：{:?}", r.dims);
        }
        Ok(Q { off: r.off, n, k, s: r.scale.context("缺少量化参数")? })
    }
    fn norm(&self, name: &str) -> Result<(Vec<f32>, Vec<f32>)> {
        Ok((self.f32(&format!("{name}.weight"), &[D])?, self.f32(&format!("{name}.bias"), &[D])?))
    }
}

/// 把 GPU 需要的权重/参数按 256B 对齐拼成一个连续 blob（libwtlgpu 要求对齐指针）
struct Blob {
    buf: Vec<u8>,
}

impl Blob {
    fn new() -> Blob {
        Blob { buf: Vec::with_capacity(160 << 20) }
    }
    fn add(&mut self, data: &[u8]) -> usize {
        while self.buf.len() % 256 != 0 {
            self.buf.push(0);
        }
        let off = self.buf.len();
        self.buf.extend_from_slice(data);
        off
    }
    fn add_f32(&mut self, v: &[f32]) -> usize {
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.add(b)
    }
    fn add_q(&mut self, l: &Loader, name: &str, n: usize, k: usize) -> Result<PlanQ> {
        self.add_q_padded(l, name, n, k, 0)
    }
    /// 尾部补 pad 行（int8 全 0）——用于把 GEMM 的 M 维凑成 4 的倍数
    fn add_q_padded(&mut self, l: &Loader, name: &str, n: usize, k: usize, pad: usize) -> Result<PlanQ> {
        let full = format!("{name}.weight_quantized_s8");
        let r = l.rec(&full)?;
        if r.dtype != I8 || r.len != n * k {
            bail!("{full} 形状不符：{:?}", r.dims);
        }
        let src = &l.d[r.off..r.off + r.len];
        let off = if pad == 0 {
            self.add(src)
        } else {
            let mut w = vec![0i8; (n + pad) * k];
            w[..n * k].copy_from_slice(unsafe { std::slice::from_raw_parts(src.as_ptr() as *const i8, src.len()) });
            self.add(unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len()) })
        };
        Ok(PlanQ { off, s: r.scale.context("缺少量化参数")? })
    }
    fn add_f32_padded(&mut self, v: &[f32], pad: usize, fill: f32) -> usize {
        if pad == 0 {
            return self.add_f32(v);
        }
        let mut x = v.to_vec();
        x.resize(v.len() + pad, fill);
        self.add_f32(&x)
    }
}

fn build_plan(l: &Loader, vocab: usize) -> Result<(Plan, Vec<u8>)> {
    let mut b = Blob::new();
    let mut layers = Vec::with_capacity(LAYERS);
    for i in 0..LAYERS {
        let p = |s: &str| format!("layers.{i}.{s}");
        let mut push = || -> Result<PlanLayer> {
            Ok(PlanLayer {
                pn_g: b.add_f32(&l.f32(&p("prior_norm.weight"), &[D])?),
                pn_b: b.add_f32(&l.f32(&p("prior_norm.bias"), &[D])?),
                mn_g: b.add_f32(&l.f32(&p("middle_norm.weight"), &[D])?),
                mn_b: b.add_f32(&l.f32(&p("middle_norm.bias"), &[D])?),
                q: b.add_q(l, &p("attention.q_conv"), D, D)?,
                k: b.add_q(l, &p("attention.k_conv"), D, D)?,
                v: b.add_q(l, &p("attention.v_conv"), D, D)?,
                o: b.add_q(l, &p("attention.output_transform_conv"), D, D)?,
                f1: b.add_q(l, &p("feed_forward.filter_conv"), FF, D)?,
                f1b: b.add_f32(&l.f32(&p("feed_forward.filter_conv.bias"), &[FF])?),
                f2: b.add_q(l, &p("feed_forward.output_conv"), D, FF)?,
                f2b: b.add_f32(&l.f32(&p("feed_forward.output_conv.bias"), &[D])?),
            })
        };
        layers.push(push()?);
    }
    let plan = Plan {
        layers,
        lin: b.add_q(l, "frontend.linear", D, FLAT)?,
        lin_b: b.add_f32(&l.f32("frontend.linear.bias", &[D])?),
        out: b.add_q_padded(l, "output_conv", vocab, D, vocab.next_multiple_of(4) - vocab)?,
        out_b: b.add_f32_padded(&l.f32("output_conv.bias", &[vocab])?, vocab.next_multiple_of(4) - vocab, -1e30),
        out_n: vocab.next_multiple_of(4),
        c1: b.add_f32(&l.f32("frontend.conv2d.weight", &[F1, 1, 5, 5])?),
        c2: b.add_f32(&l.f32("frontend.conv2d_1.weight", &[F2, F1, 7, 7])?),
        c3: b.add_f32(&l.f32("frontend.conv2d_2.weight", &[F3, F2, 3, 3])?),
    };
    Ok((plan, b.buf))
}

impl Model {
    pub fn load(dir: &Path) -> Result<Model> {
        let xnet = std::fs::read_dir(dir)
            .map_err(|_| anyhow::anyhow!("离线模型未下载"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().is_some_and(|e| e == "xnet"))
            .context("模型文件不存在")?;
        let map = unsafe { Mmap::map(&std::fs::File::open(&xnet)?)? };
        #[cfg(unix)]
        let _ = map.advise(memmap2::Advice::WillNeed);
        let l = Loader { d: &map, r: xnet::parse(&map)? };
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = |s: &str| format!("layers.{i}.{s}");
            layers.push(Layer {
                pn: l.norm(&p("prior_norm"))?,
                mn: l.norm(&p("middle_norm"))?,
                q: l.q(&p("attention.q_conv"), D, D)?,
                k: l.q(&p("attention.k_conv"), D, D)?,
                v: l.q(&p("attention.v_conv"), D, D)?,
                o: l.q(&p("attention.output_transform_conv"), D, D)?,
                f1: l.q(&p("feed_forward.filter_conv"), FF, D)?,
                f1b: l.f32(&p("feed_forward.filter_conv.bias"), &[FF])?,
                f2: l.q(&p("feed_forward.output_conv"), D, FF)?,
                f2b: l.f32(&p("feed_forward.output_conv.bias"), &[D])?,
            });
        }
        let out = l.rec("output_conv.bias")?.dims.first().copied().unwrap_or(0);
        let vocab: Vec<String> = std::fs::read_to_string(dir.join("dict.decoder.utf8.txt"))
            .context("词表不存在")?
            .lines()
            .map(|s| s.split(' ').nth(1).unwrap_or("").to_string())
            .collect();
        if vocab.len() != out {
            bail!("词表大小 {} 与模型输出 {out} 不符", vocab.len());
        }
        // GPU 后端：一次性把权重拼成对齐 blob 并上传（135MB，约 30~60ms）
        let (gpu, plan) = if std::env::var("WTL_GPU").is_ok() {
            let (plan, blob) = build_plan(&l, out)?;
            let g = Gpu::new(&blob, TMAX, D, FF, plan.out_n, FLAT, HEADS);
            (Some(g), Some(plan))
        } else {
            (None, None)
        };
        Ok(Model {
            c1: l.f32("frontend.conv2d.weight", &[F1, 1, 5, 5])?,
            c2: l.f32("frontend.conv2d_1.weight", &[F2, F1, 7, 7])?,
            c3: l.f32("frontend.conv2d_2.weight", &[F3, F2, 3, 3])?,
            lin: l.q("frontend.linear", D, FLAT)?,
            lin_b: l.f32("frontend.linear.bias", &[D])?,
            layers,
            out: l.q("output_conv", out, D)?,
            out_b: l.f32("output_conv.bias", &[out])?,
            vocab,
            map,
            gpu,
            plan,
        })
    }

    pub fn gpu_enabled(&self) -> bool {
        self.gpu.is_some()
    }

    fn mm(&self, x: &QAct, q: &Q, b: Option<&[f32]>) -> Vec<f32> {
        let w = &self.map[q.off..q.off + q.n * q.k];
        let w = unsafe { std::slice::from_raw_parts(w.as_ptr() as *const i8, w.len()) };
        qgemm(x, w, q.n, q.s, b)
    }

    /// 输入归一化特征帧，输出每个 50ms 输出帧的 argmax token id
    pub fn forward(&self, x: &[[f32; DIM]]) -> Vec<u32> {
        if self.gpu.is_some() {
            // GPU 路径：前端仍用全核 rayon（不在 4 线程小池里），GPU 部分异步发起
            return self.run_gpu(x);
        }
        // 独立小线程池：4 线程以上收益很小，却会明显增加调度开销与整机占用
        static POOL: std::sync::LazyLock<rayon::ThreadPool> = std::sync::LazyLock::new(|| {
            // 线程上限可用 WTL_THREADS 覆盖（默认仍为 4，保持与原实现一致）
            let avail = std::thread::available_parallelism().map_or(2, |n| n.get());
            let n = std::env::var("WTL_THREADS").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(avail.min(4)).clamp(1, avail.max(1));
            rayon::ThreadPoolBuilder::new().num_threads(n).thread_name(|i| format!("wtlocal-{i}")).build().unwrap()
        });
        POOL.install(|| self.run(x))
    }

    /// 卷积前端：[t][40] 特征帧 → ([t1][FLAT] 展开特征, t1)
    fn frontend(&self, x: &[[f32; DIM]]) -> Option<(usize, Vec<f32>)> {
        let t = x.len();
        if t < 5 {
            return None;
        }
        let t1 = (t - 5) / 5 + 1;
        let at = |i: usize, d: isize| -> Option<usize> { i.checked_add_signed(d) };
        let tf0 = std::time::Instant::now();

        // conv 5×5, 1→32, 时频各补 2
        let mut y1 = vec![0f32; t * F1 * DIM];
        y1.par_chunks_mut(F1 * DIM).enumerate().for_each(|(ti, y)| {
            let mut p = [[0f32; 25]; DIM];
            for (f, pf) in p.iter_mut().enumerate() {
                for i in 0..5 {
                    let Some(tt) = at(ti, i as isize - 2).filter(|&v| v < t) else { continue };
                    for j in 0..5 {
                        if let Some(ff) = at(f, j as isize - 2).filter(|&v| v < DIM) {
                            pf[i * 5 + j] = x[tt][ff];
                        }
                    }
                }
            }
            for o in 0..F1 {
                let w = &self.c1[o * 25..o * 25 + 25];
                for f in 0..DIM {
                    y[o * DIM + f] = dotf(w, &p[f]).max(0.0);
                }
            }
        });

        // conv 7×7, 32→64, 时间步长 5（时补 1、频补 3）
        let tf1 = std::time::Instant::now();
        let mut y2 = vec![0f32; t1 * F2 * W2];
        y2.par_chunks_mut(F2 * W2).enumerate().for_each(|(ti, y)| {
            let mut p = vec![0f32; DIM * K2];
            for i in 0..7 {
                let Some(tt) = at(ti * 5 + i, -1).filter(|&v| v < t) else { continue };
                let row = &y1[tt * F1 * DIM..(tt + 1) * F1 * DIM];
                for c in 0..F1 {
                    for f in 0..DIM {
                        for j in 0..7 {
                            if let Some(ff) = at(f + j, -3).filter(|&v| v < DIM) {
                                p[f * K2 + c * 49 + i * 7 + j] = row[c * DIM + ff];
                            }
                        }
                    }
                }
            }
            for o in 0..F2 {
                let w = &self.c2[o * K2..(o + 1) * K2];
                for f in 0..DIM {
                    y[o * W2 + f] = dotf(w, &p[f * K2..(f + 1) * K2]).max(0.0);
                }
            }
        });

        // conv 3×3, 64→192, 频率步长 2（时补 1）→ [t1][192×20]
        let tf2 = std::time::Instant::now();
        let mut y3 = vec![0f32; t1 * FLAT];
        y3.par_chunks_mut(FLAT).enumerate().for_each(|(ti, y)| {
            let mut p = vec![0f32; 20 * K3];
            for i in 0..3 {
                let Some(tt) = at(ti + i, -1).filter(|&v| v < t1) else { continue };
                let row = &y2[tt * F2 * W2..(tt + 1) * F2 * W2];
                for c in 0..F2 {
                    for f in 0..20 {
                        for j in 0..3 {
                            p[f * K3 + c * 9 + i * 3 + j] = row[c * W2 + 2 * f + j];
                        }
                    }
                }
            }
            for o in 0..F3 {
                let w = &self.c3[o * K3..(o + 1) * K3];
                for f in 0..20 {
                    y[o * 20 + f] = dotf(w, &p[f * K3..(f + 1) * K3]).max(0.0);
                }
            }
        });
        if std::env::var("WTL_GPU_PROF").is_ok() {
            eprintln!(
                "  [front] t={t} t1={t1} conv1={:5.2}ms conv2={:5.2}ms conv3={:5.2}ms",
                tf0.elapsed().as_secs_f64() * 1e3 - (tf2 - tf1).as_secs_f64() * 1e3 - tf2.elapsed().as_secs_f64() * 1e3,
                (tf2 - tf1).as_secs_f64() * 1e3,
                tf2.elapsed().as_secs_f64() * 1e3
            );
        }
        Some((t1, y3))
    }

    fn run(&self, x: &[[f32; DIM]]) -> Vec<u32> {
        match self.frontend(x) {
            Some((t1, y3)) => self.run_post(t1, y3),
            None => Vec::new(),
        }
    }

    /// 前端之后：linear 投影 + 40 层 + CTC（CPU 路径）
    fn run_post(&self, t1: usize, y3: Vec<f32>) -> Vec<u32> {
        let mut h = self.mm(&quant(&y3, FLAT), &self.lin, Some(&self.lin_b));
        for l in &self.layers {
            let y = quant(&norm(&h, &l.pn), D);
            let (q, k, v) = (self.mm(&y, &l.q, None), self.mm(&y, &l.k, None), self.mm(&y, &l.v, None));
            add(&mut h, &self.mm(&quant(&attention(&q, &k, &v, t1), D), &l.o, None));
            let mut f = self.mm(&quant(&norm(&h, &l.mn), D), &l.f1, Some(&l.f1b));
            f.par_iter_mut().for_each(|v| *v = gelu(*v));
            add(&mut h, &self.mm(&quant(&f, FF), &l.f2, Some(&l.f2b)));
        }
        let lo = self.mm(&quant(&h, D), &self.out, Some(&self.out_b));
        lo.par_chunks(self.out.n)
            .map(|r| r.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32)
            .collect()
    }

    /// GPU 路径：卷积前端 + 40 层 + 输出层全在 GPU 上跑，只回 t1 个 token id
    fn run_gpu(&self, x: &[[f32; DIM]]) -> Vec<u32> {
        let prof = std::env::var("WTL_GPU_PROF").is_ok();
        let t_all = std::time::Instant::now();
        let t = x.len();
        if t < 5 {
            return Vec::new();
        }
        let t1 = (t - 5) / 5 + 1;
        let g = self.gpu.as_ref().unwrap();
        let pl = self.plan.as_ref().unwrap();
        if t1 > g.tmax || pl.out_n > g.vocab {
            return match self.frontend(x) {
                Some((t1, y3)) => self.run_post(t1, y3),
                None => Vec::new(),
            };
        }
        // 前端：默认整段搬 GPU（WTL_GPU_FRONT=0 回退成 CPU 前端 + 只上传量化后的 y3）
        let gpu_front = std::env::var("WTL_GPU_FRONT").map(|v| v != "0").unwrap_or(true) && t <= g.tmax_feat;
        if gpu_front {
            g.upload_feats(x, t);
            g.front(t, t1, pl.c1, pl.c2, pl.c3);
            g.quant_y3(t1, FLAT);
        } else {
            let Some((_, y3)) = self.frontend(x) else { return Vec::new() };
            let yq = quant(&y3, FLAT);
            g.upload_y3(&yq.q, &yq.s);
        }
        let t_front = t_all.elapsed();
        g.gemm(g.y3q, g.ys, t1, FLAT, pl.lin.off, D, pl.lin.s, pl.lin_b, g.h);
        let tl = std::time::Instant::now();
        for l in &pl.layers {
            g.norm_quant(t1, D, l.pn_g, l.pn_b, g.xq, g.xs);
            g.gemm(g.xq, g.xs, t1, D, l.q.off, D, l.q.s, NO_BIAS, g.q);
            g.gemm(g.xq, g.xs, t1, D, l.k.off, D, l.k.s, NO_BIAS, g.k);
            g.gemm(g.xq, g.xs, t1, D, l.v.off, D, l.v.s, NO_BIAS, g.v);
            g.attention(t1, D, HEADS, g.q, g.k, g.v, g.att);
            g.quant(g.att, t1, D, g.aq, g.asc);
            g.gemm(g.aq, g.asc, t1, D, l.o.off, D, l.o.s, NO_BIAS, g.o);
            // h += o，然后立刻对 h 做下一层的 norm+quant（融合成一个 kernel）
            g.add_norm_quant(t1, D, l.mn_g, l.mn_b, g.xq, g.xs);
            g.gemm(g.xq, g.xs, t1, D, l.f1.off, FF, l.f1.s, l.f1b, g.f);
            g.gelu_quant(t1, FF, g.fq, g.fsc);
            g.gemm(g.fq, g.fsc, t1, FF, l.f2.off, D, l.f2.s, l.f2b, g.o);
            g.add(g.h, g.o, t1, D);
        }
        if prof {
            g.sync();
        }
        let t_layers = tl.elapsed();
        let to = std::time::Instant::now();
        g.quant(g.h, t1, D, g.aq, g.asc);
        g.gemm(g.aq, g.asc, t1, D, pl.out.off, pl.out_n, pl.out.s, pl.out_b, g.lo);
        g.argmax(g.lo, t1, pl.out_n, g.ids);
        g.sync();
        let ids = g.download_ids(t1);
        if prof {
            eprintln!(
                "[gpu] t={t:4} t1={t1:4} gpu_front={:5} front={:6.2}ms layers={:6.2}ms out={:5.2}ms total={:6.2}ms",
                gpu_front,
                t_front.as_secs_f64() * 1e3,
                t_layers.as_secs_f64() * 1e3,
                to.elapsed().as_secs_f64() * 1e3,
                t_all.elapsed().as_secs_f64() * 1e3
            );
        }
        ids
    }

    /// CTC 折叠：去重、去 blank；英文词首（|xxx）前补空格
    pub fn text(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        let mut prev = u32::MAX;
        for &i in ids {
            if i != prev && i != 0 {
                match self.vocab[i as usize].as_str() {
                    "<UNK>" => {}
                    "<SPACE>" => s.push(' '),
                    t if t.len() > 1 && t.starts_with('|') => {
                        if s.ends_with(|c: char| c.is_ascii_alphanumeric()) {
                            s.push(' ');
                        }
                        s.push_str(&t[1..]);
                    }
                    t => s.push_str(t),
                }
            }
            prev = i;
        }
        s
    }
}

fn norm(x: &[f32], (g, b): &(Vec<f32>, Vec<f32>)) -> Vec<f32> {
    let mut y = x.to_vec();
    y.par_chunks_mut(D).for_each(|r| {
        let m = r.iter().sum::<f32>() / D as f32;
        let v = r.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / D as f32;
        let inv = 1.0 / (v + 1e-6).sqrt();
        for i in 0..D {
            r[i] = (r[i] - m) * inv * g[i] + b[i];
        }
    });
    y
}

fn add(h: &mut [f32], o: &[f32]) {
    h.iter_mut().zip(o).for_each(|(a, b)| *a += b);
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}

fn attention(q: &[f32], k: &[f32], v: &[f32], t: usize) -> Vec<f32> {
    let mut out = vec![0f32; t * D];
    out.par_chunks_mut(D).enumerate().for_each(|(i, o)| {
        let mut s = vec![0f32; t];
        for h in 0..HEADS {
            let qi = &q[i * D + h * HD..i * D + (h + 1) * HD];
            for (j, sj) in s.iter_mut().enumerate() {
                *sj = dotf(qi, &k[j * D + h * HD..j * D + (h + 1) * HD]) * 0.125;
            }
            let m = s.iter().fold(f32::MIN, |a, &b| a.max(b));
            let mut z = 0.0;
            for sj in s.iter_mut() {
                *sj = (*sj - m).exp();
                z += *sj;
            }
            let oh = &mut o[h * HD..(h + 1) * HD];
            for (j, &a) in s.iter().enumerate() {
                let vj = &v[j * D + h * HD..j * D + (h + 1) * HD];
                for d in 0..HD {
                    oh[d] += a / z * vj[d];
                }
            }
        }
    });
    out
}
