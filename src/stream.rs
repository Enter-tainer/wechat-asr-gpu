//! 流式识别：说话中分块解码出中间结果（每块带前文/后文上下文重算，已定稿块不再变化）；
//! 结束时整段一次解码出定稿，整段更准；太长时注意力开销平方增长，改用已定稿块 + 尾部
//!
//! 批模式（不需要中间结果）下，块长直接决定吞吐：每块的上下文开销 = prev + post + pad，
//! 有效计算量 ≈ (prev + blk + post + pad) / blk。默认 400 帧时是 1.87x 重算放大，
//! 批模式建议 --blk 1600（≈1.22x）。可用环境变量 WTL_PREV / WTL_BLK / WTL_POST 覆盖。
use crate::fbank::{Fbank, DIM};
use crate::model::Model;
use std::sync::Arc;

/// 默认参数（特征帧，10ms/帧）
const PREV_DEFAULT: usize = 200;
const BLK_DEFAULT: usize = 400;
const POST_DEFAULT: usize = 100;
/// 每个解码窗口末尾补的静音帧：窗口在语音中途截断时，模型常整窗不出字，补静音后稳定
const PAD: usize = 50;
/// 整段解码的上限（特征帧）：20 秒内定稿等待约 0.5 秒，70 秒要近 4 秒
const WHOLE_MAX: usize = 2000;
/// 超过这么多特征帧（=200 秒）才值得用 CUDA graph 舍入捕获开销；可用 WTL_GRAPH=0/1 强制
const GRAPH_MIN_FRAMES: usize = 20000;

pub struct Cfg {
    pub prev: usize,
    pub blk: usize,
    pub post: usize,
}

impl Cfg {
    fn new() -> Cfg {
        let get = |k: &str, d: usize| {
            std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok()).filter(|&v| v >= 100).unwrap_or(d)
        };
        Cfg { prev: get("WTL_PREV", PREV_DEFAULT), blk: get("WTL_BLK", BLK_DEFAULT), post: get("WTL_POST", POST_DEFAULT) }
    }
}

pub struct Stream {
    m: Arc<Model>,
    fb: Fbank,
    feats: Vec<[f32; DIM]>,
    ids: Vec<u32>,
    s: usize,
    fresh: usize,
    cfg: Cfg,
}

impl Stream {
    pub fn new(m: Arc<Model>) -> Self {
        m.set_graph_on(false);
        Stream { m, fb: Fbank::new(), feats: Vec::new(), ids: Vec::new(), s: 0, fresh: 0, cfg: Cfg::new() }
    }

    /// 当前生效的解码参数（供 CLI 打印）
    pub fn cfg(&self) -> &Cfg {
        &self.cfg
    }

    pub fn push(&mut self, pcm: &[i16]) {
        let n = self.feats.len();
        self.fb.push(pcm, &mut self.feats);
        // 批模式整段喂入时这里就知道总长度：超过阈值才启用 CUDA graph
        // （短音频捕获/实例化开销 > 省下的 host 入队时间）
        if self.feats.len() >= GRAPH_MIN_FRAMES {
            self.m.set_graph_on(true);
        }
        self.fresh += self.feats.len() - n;
        while self.feats.len() >= self.s + self.cfg.blk + self.cfg.post {
            self.commit();
        }
    }

    /// 只跑 fbank、不做分块解码。批窗口模式（`--mlx --batch N`）用：
    /// 整个文件先出完特征，再按固定网格一次性批算。
    pub fn push_raw(&mut self, pcm: &[i16]) {
        self.fb.push(pcm, &mut self.feats);
        self.fresh = self.feats.len();
    }

    /// 批窗口整段解码（MLX 后端）：固定网格 + B 个窗口一批 + 每窗只取自己的 [s, s+blk) 区间。
    ///
    /// 为什么可以批：这是 CTC 编码器，窗口之间没有模型状态依赖（上下文只来自音频 prev/post 重叠），
    /// 所以 B 个窗口堆成一个 batch 后 GEMM 从 (390,512) 变成 (B*390,512)，
    /// M2 Pro 上 fp16 利用率从 ~11% 提到 ~45%（bv58 254ms → 170ms）。
    #[cfg(feature = "mlx")]
    pub fn finish_mlx_batched(self, batch: usize) -> String {
        let n = self.feats.len();
        if n == 0 {
            return String::new();
        }
        if n <= WHOLE_MAX {
            // 与 Rust 原有的整段解码一致：不补 pad、不加左侧上下文
            return tidy(&self.m.text(&self.m.forward(&self.feats)));
        }
        let (prev, blk, post) = (self.cfg.prev, self.cfg.blk, self.cfg.post);
        let sil = self.fb.silence();
        let head = prev / 5;
        let batch = batch.max(1);
        let starts: Vec<usize> = (0..n).step_by(blk).collect();
        let full: Vec<usize> = starts.iter().copied().filter(|&s| s + blk <= n).collect();
        let tail: Vec<usize> = starts.iter().copied().filter(|&s| s + blk > n).collect();
        let mut groups: Vec<Vec<usize>> = full.chunks(batch).map(|c| c.to_vec()).collect();
        if !tail.is_empty() {
            groups.push(tail);
        }
        let mut ids: Vec<u32> = Vec::new();
        for grp in groups {
            let tail_grp = grp[0] + blk > n;
            let wins: Vec<Vec<[f32; DIM]>> =
                grp.iter().map(|&s0| window(&self.feats, &sil, s0, prev, blk, post, n, !tail_grp)).collect();
            let t = wins[0].len();
            for (&s0, out) in grp.iter().zip(self.m.forward_batch(&wins, t).iter()) {
                let avail = blk.min(n - s0);
                // 最后一个窗口多收 50 帧 pad 的 id，否则末尾半个字会被截掉
                let keep = avail / 5 + if tail_grp { PAD / 5 } else { 0 };
                let hi = (head + keep).min(out.len());
                ids.extend_from_slice(&out[head.min(hi)..hi]);
            }
        }
        tidy(&self.m.text(&ids))
    }

    /// 距上次中间结果新增的特征帧数
    pub fn fresh(&self) -> usize {
        self.fresh
    }

    /// 解码 [s - prev, end) 并补 pad 帧静音，返回 s 之后的输出帧
    fn decode(&self, prev: usize, end: usize, pad: usize) -> Vec<u32> {
        let a = self.s.saturating_sub(prev);
        let mut x = self.feats[a..end].to_vec();
        x.resize(x.len() + pad, self.fb.silence());
        let out = self.m.forward(&x);
        out.get((self.s - a) / 5..).unwrap_or_default().to_vec()
    }

    fn commit(&mut self) {
        let blk = self.cfg.blk;
        let ids = self.decode(self.cfg.prev, self.s + blk + self.cfg.post, PAD);
        let mut ids = ids;
        ids.truncate(blk / 5);
        // 在块尾附近的连续 blank 处截断，避免字恰好落在块边界上被两边都丢掉
        if let Some(c) = (ids.len() / 2..ids.len()).rev().find(|&c| ids[c] == 0 && ids[c - 1] == 0) {
            ids.truncate(c);
        }
        self.s += ids.len() * 5;
        self.ids.extend(ids);
    }

    /// 中间结果：已定稿部分 + 尾部临时解码（前文减半以省算力）；去掉末尾标点，避免边说边打时句号反复增删
    pub fn partial(&mut self) -> String {
        self.fresh = 0;
        let mut ids = self.ids.clone();
        if self.feats.len() > self.s {
            ids.extend(self.decode(self.cfg.prev / 2, self.feats.len(), PAD));
        }
        tidy(&self.m.text(&ids)).trim_end_matches(PUNCT).to_string()
    }

    pub fn finish(mut self) -> String {
        if self.feats.len() <= WHOLE_MAX {
            return tidy(&self.m.text(&self.m.forward(&self.feats)));
        }
        if self.feats.len() > self.s {
            let tail = self.decode(self.cfg.prev, self.feats.len(), PAD);
            self.ids.extend(tail);
        }
        tidy(&self.m.text(&self.ids))
    }
}

const PUNCT: &[char] = &['，', '。', '？', '！', '、', ',', '.', '?', '!'];

/// 拼一个固定网格窗口：[s0-prev, s0+blk+post) + 末尾 PAD 帧静音；左右不足补静音。
/// full=true 时补齐到 prev+blk+post（同批窗口必须等长），false 给最后一个窗口按真实长度收尾。
#[cfg(feature = "mlx")]
fn window(
    feats: &[[f32; DIM]],
    sil: &[f32; DIM],
    s0: usize,
    prev: usize,
    blk: usize,
    post: usize,
    n: usize,
    full: bool,
) -> Vec<[f32; DIM]> {
    let left = prev.min(s0);
    let a = s0 - left;
    let b = (s0 + blk + post).min(n);
    let t_win = prev + blk + post;
    let mut x: Vec<[f32; DIM]> = Vec::with_capacity(t_win + PAD);
    for _ in 0..(prev - left) {
        x.push(*sil);
    }
    x.extend_from_slice(&feats[a..b]);
    if full {
        while x.len() < t_win {
            x.push(*sil);
        }
    }
    for _ in 0..PAD {
        x.push(*sil);
    }
    x
}

/// 开口前的底噪偶尔被识别成一个逗号；偶尔出现「，。」这样的连续标点，只留后一个
fn tidy(t: &str) -> String {
    let c: Vec<char> = t.trim_start_matches(PUNCT).chars().collect();
    c.iter().enumerate().filter(|&(i, ch)| !(PUNCT.contains(ch) && c.get(i + 1).is_some_and(|n| PUNCT.contains(n)))).map(|(_, ch)| ch).collect()
}
