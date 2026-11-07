//! M2 Pro CPU int8 dot 微基准：复刻 wechat-asr-gpu/src/kernels.rs 的 dot2x4 结构
//! （2 行激活 x 4 行权重，vdotq_s32，16B/步），测 GMAC/s，再换算成 ASR 的「倍实时」。
//! 编译： rustc +nightly -O analysis/bench_cpu_neon.rs -o /tmp/b && /tmp/b
//! 注意：vdotq_s32 在 stable 上是 unstable（E0658, stdarch_neon_dotprod），
//!       所以 wechat-asr-gpu 的 NEON 路径在 Mac 上必须用 nightly 才编得过。
#![feature(stdarch_neon_dotprod)]
use std::arch::aarch64::*;
use std::time::Instant;

const K: usize = 512; // D=512
const N: usize = 2048; // FF=2048
const T: usize = 1600; // blk=1600 帧
const REPS: usize = 12;

#[inline]
unsafe fn dot2x4(x: [*const i8; 2], w: [*const i8; 4], k: usize) -> [[i32; 4]; 2] {
    let mut a = [vdupq_n_s32(0); 8];
    let mut i = 0;
    while i < k {
        let x0 = vld1q_s8(x[0].add(i));
        let x1 = vld1q_s8(x[1].add(i));
        for b in 0..4 {
            let wv = vld1q_s8(w[b].add(i));
            a[b] = vdotq_s32(a[b], x0, wv);
            a[4 + b] = vdotq_s32(a[4 + b], x1, wv);
        }
        i += 16;
    }
    let mut r = [[0; 4]; 2];
    for b in 0..4 {
        r[0][b] = vaddvq_s32(a[b]);
        r[1][b] = vaddvq_s32(a[4 + b]);
    }
    r
}

/// 一个 GEMM：out[t][n] = x[t][k] . w[n][k]（int32 累加），按 n 分块
unsafe fn gemm(x: &[i8], w: &[i8], t: usize, n: usize, k: usize, sink: &mut i64) {
    let mut j = 0;
    while j < n {
        let rows = (n - j).min(4);
        let wp: [*const i8; 4] = std::array::from_fn(|b| w.as_ptr().add((j + b.min(rows - 1)) * k));
        let mut i = 0;
        while i < t {
            let r2 = (t - i).min(2);
            let xp = [x.as_ptr().add(i * k), x.as_ptr().add((i + r2 - 1) * k)];
            let acc = dot2x4(xp, wp, k);
            *sink += acc[0][0] as i64 + acc[1][rows - 1] as i64;
            i += r2;
        }
        j += rows;
    }
}

fn main() {
    unsafe {
        assert!(std::arch::is_aarch64_feature_detected!("dotprod"), "no dotprod");
        let x: Vec<i8> = (0..T * K).map(|i| (i.wrapping_mul(2654435761) >> 13) as i8).collect();
        let w: Vec<i8> = (0..N * K).map(|i| (i.wrapping_mul(40503) >> 11) as i8).collect();
        let macs = (T * N * K) as f64; // 每个 GEMM 的 MAC 数

        // 单线程
        let t0 = Instant::now();
        let mut sink = 0i64;
        for _ in 0..REPS {
            gemm(&x, &w, T, N, K, &mut sink);
        }
        let d1 = t0.elapsed().as_secs_f64();
        println!("1 thread : {:.1} GMAC/s  ({:.3} s/GEMM)", macs * REPS as f64 / d1 / 1e9, d1 / REPS as f64);

        // 多线程（按 n 分片，模拟 rayon）
        for th in [4usize, 6, 8, 10] {
            let mut tot = 0i64;
            let t0 = Instant::now();
            for _ in 0..REPS {
                let nt = N / th;
                let mut hs = Vec::new();
                for b in 0..th {
                    let xp = x.as_ptr() as usize;
                    let wp = w.as_ptr() as usize;
                    let lo = b * nt;
                    let hi = if b == th - 1 { N } else { lo + nt };
                    hs.push(std::thread::spawn(move || {
                        let x = std::slice::from_raw_parts(xp as *const i8, T * K);
                        let w = std::slice::from_raw_parts(wp as *const i8, N * K);
                        let w = &w[lo * K..];
                        let mut s = 0i64;
                        unsafe { gemm(&x, w, T, hi - lo, K, &mut s) };
                        s
                    }));
                }
                for h in hs {
                    tot += h.join().unwrap();
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            let gm = macs * REPS as f64 / dt / 1e9;
            println!(
                "{th:2} threads: {gm:.1} GMAC/s  ({:.3} ms/GEMM)  => ASR ~{:.1}x realtime (23.35 GMAC/audio-s)  [sink {}]",
                dt / REPS as f64 * 1e3,
                gm / 23.35,
                tot + sink
            );
        }
    }
}
