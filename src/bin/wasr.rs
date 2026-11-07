//! wasr —— 微信输入法离线 ASR 引擎（wtlocal / .xnet）的批模式 CLI，CPU / GPU 双后端。
//!
//! 用法：
//!   wasr <模型目录> [选项] <a.wav> [b.wav ...]
//!
//! 选项：
//!   --gpu            用 GPU 后端（需要 NVIDIA GPU + CUDA 编译的 libwasr_gpu.so）
//!   --cpu            强制 CPU 后端（对照用）
//!   --threads N      CPU 线程上限（默认 8；实测 8 是拐点）
//!   --blk N          解码块长（特征帧，10ms/帧）：默认 1600（批模式最优），流式才用 400
//!   --json FILE      逐文件结果写 JSON
//!   --txt DIR        每条音频的转录另存为 DIR/<name>.txt
//!   --unpack PKG DIR 解包官方语音包（.apk）到 DIR，得到模型文件
//!
//! stdout 为 Tab 分隔：`名字  音频秒  解码毫秒  RTF  文本`；stderr 打加载信息与批聚合吞吐。
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use wechat_asr_gpu::{load, pack, Stream};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        eprintln!(
            "用法: wasr <模型目录> [--gpu] [--cpu] [--threads N] [--blk N] [--json FILE] [--txt DIR] <a.wav> [b.wav ...]\n\
             解包官方语音包: wasr --unpack <pak.apk> <模型输出目录>"
        );
        std::process::exit(2);
    }
    if args[0] == "--unpack" {
        let t = Instant::now();
        pack::unpack(args[1].as_ref(), args[2].as_ref()).unwrap();
        println!("unpacked in {:.1}s", t.elapsed().as_secs_f64());
        return;
    }

    let dir = PathBuf::from(&args[0]);
    let mut files: Vec<String> = Vec::new();
    let (mut json_out, mut txt_dir) = (None::<String>, None::<String>);
    let mut gpu = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--gpu" => gpu = true,
            "--cpu" => gpu = false,
            "--threads" => {
                i += 1;
                std::env::set_var("WTL_THREADS", &args[i]);
            }
            "--blk" => {
                i += 1;
                std::env::set_var("WTL_BLK", &args[i]);
            }
            "--json" => {
                i += 1;
                json_out = Some(args[i].clone());
            }
            "--txt" => {
                i += 1;
                txt_dir = Some(args[i].clone());
            }
            other => files.push(other.to_string()),
        }
        i += 1;
    }
    if gpu {
        std::env::set_var("WTL_GPU", "1");
    }
    if std::env::var("WTL_BLK").is_err() {
        std::env::set_var("WTL_BLK", "1600"); // 批模式默认块长
    }

    let t0 = Instant::now();
    let m = match load(&dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("加载模型失败: {e:#}\n（先跑 `wasr --unpack <微信语音包.apk> <dir>` 准备模型目录）");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "load {load_ms:.0}ms  backend={}  threads={}  blk={}",
        if m.gpu_enabled() { "GPU" } else { "CPU" },
        std::env::var("WTL_THREADS").unwrap_or_else(|_| "8".into()),
        std::env::var("WTL_BLK").unwrap_or_else(|_| "1600".into())
    );
    if files.is_empty() {
        eprintln!("没有输入音频");
        std::process::exit(2);
    }

    let mut rows = Vec::new();
    let batch = Instant::now();
    let (mut total_audio, mut total_ms) = (0.0f64, 0.0f64);
    for f in &files {
        let name = Path::new(f).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let pcm: Vec<i16> = match hound::WavReader::open(f) {
            Ok(r) => r.into_samples::<i16>().map(|s| s.unwrap_or(0)).collect(),
            Err(e) => {
                eprintln!("跳过 {f}: {e}");
                continue;
            }
        };
        let audio = pcm.len() as f64 / 16000.0;
        let t = Instant::now();
        let mut s = Stream::new(m.clone());
        for c in pcm.chunks(320) {
            s.push(c);
        }
        let text = s.finish();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        total_audio += audio;
        total_ms += ms;
        println!("{name}\t{audio:.2}\t{ms:.0}\t{:.4}\t{text}", ms / 1000.0 / audio);
        if let Some(d) = &txt_dir {
            let _ = std::fs::create_dir_all(d);
            let _ = std::fs::write(Path::new(d).join(format!("{name}.txt")), &text);
        }
        rows.push((name, audio, ms, ms / 1000.0 / audio, text));
    }
    let wall = batch.elapsed().as_secs_f64();
    eprintln!(
        "\n批模式: {} 条, 音频 {:.1}s, 解码合计 {:.2}s, 批墙钟 {:.2}s, 聚合 {:.1}x 实时, 平均 RTF {:.4}",
        rows.len(),
        total_audio,
        total_ms / 1e3,
        wall,
        total_audio / wall,
        total_ms / 1e3 / total_audio
    );

    if let Some(p) = json_out {
        let mut f = std::fs::File::create(&p).expect("create json");
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ");
        writeln!(f, "{{").unwrap();
        writeln!(
            f,
            "  \"backend\": \"{}\", \"threads\": {}, \"blk\": {}, \"load_ms\": {:.0},",
            if m.gpu_enabled() { "gpu" } else { "cpu" },
            std::env::var("WTL_THREADS").unwrap_or_else(|_| "8".into()),
            std::env::var("WTL_BLK").unwrap_or_else(|_| "1600".into()),
            load_ms
        )
        .unwrap();
        writeln!(
            f,
            "  \"total_audio_s\": {:.2}, \"total_decode_s\": {:.2}, \"batch_wall_s\": {:.2}, \"agg_x_realtime\": {:.1},",
            total_audio,
            total_ms / 1e3,
            wall,
            total_audio / wall
        )
        .unwrap();
        writeln!(f, "  \"files\": [").unwrap();
        for (k, (name, audio, ms, rtf, text)) in rows.iter().enumerate() {
            writeln!(
                f,
                "    {{\"name\": \"{}\", \"audio_s\": {:.2}, \"decode_ms\": {:.0}, \"rtf\": {:.4}, \"text\": \"{}\"}}{}",
                name,
                audio,
                ms,
                rtf,
                esc(text),
                if k + 1 == rows.len() { "" } else { "," }
            )
            .unwrap();
        }
        writeln!(f, "  ]").unwrap();
        writeln!(f, "}}").unwrap();
        eprintln!("JSON 已写入 {p}");
    }
}
