# autosubs

**Automatic, GPU-accelerated SRT subtitle generation from video files.**

Point it at a video file or a whole series folder and it writes a synced `.srt`
next to every video — no interaction, no temp files:

```
autosubs "E:\Downloads\Planetes\Season 01"
```

Powered by [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with the
**large-v3** model on CUDA, tuned for quality *and* speed (≈15–20x realtime on
an RTX 3070; a 25-minute episode takes about 90 seconds).

## Features

- **Fully automatic** — scans a file or directory tree (mp4, mkv, avi, mov,
  webm, wmv, ts, …), skips videos that already have `.srt` files, writes
  subtitles next to the video
- **Quality-focused decoding** — large-v3, beam search 5, flash attention,
  language auto-detection, temperature fallback
- **Loop prevention at three levels**
  1. a small [patch](patches/whisper-cpp-repetition-ratio.patch) to whisper.cpp
     implementing OpenAI's documented `compression_ratio_threshold` window
     rejection (loops are retried up the temperature ladder instead of being
     emitted),
  2. silero-VAD speech blocks (≤180 s) transcribed with fresh decoder context
     per block — the documented `condition_on_previous_text=False` mitigation,
     without losing punctuation/style *inside* a block, plus a surgical
     fresh-re-transcribe repair for cyclic loops,
  3. deterministic post-filters (identical/cyclic collapse, cue-duration caps,
     speech-rate sanity)
- **Fast audio path** — ffmpeg piped straight into memory (16 kHz mono f32),
  50 Hz high-pass + static loudness conditioning; opt-in RNNoise denoise
- **Correct SRT output** — overlap/duration repair, UTF-8, proper Windows wide-
  API path handling

## Usage

```
autosubs <file-or-directory> [more...] [options]

  -m, --model <path>       ggml model (default: models/ggml-large-v3.bin)
  -l, --language <code>    spoken language, or 'auto' (default: auto)
      --translate          translate to English
  -o, --output-dir <dir>   write .srt files elsewhere
      --audio-stream <n>   audio stream index (default: 0)
      --beam <n>           beam size, 1 = greedy (default: 5)
      --chunk <sec>        VAD block length (default: 180)
      --prompt <text>      initial prompt (names, jargon)
      --denoise <model>    RNNoise .rnnn model (opt-in)
      --no-vad / --no-gpu / --no-flash-attn
      -f, --force          regenerate existing subtitles
      -n, --dry-run        list what would be done
      -v, --verbose        stream segments while transcribing
```

## Building

Requirements: Windows, CMake + Ninja, Visual Studio (MSVC), CUDA toolkit, git.

```bat
build_autosubs.bat          REM clones + patches whisper.cpp, builds CUDA dlls
powershell -ExecutionPolicy Bypass -File make_release.ps1   REM portable release zip
```

`build_autosubs.bat` clones whisper.cpp v1.9.5 and applies
`patches/whisper-cpp-repetition-ratio.patch` automatically. The CUDA build
targets `sm_86` by default — change `CMAKE_CUDA_ARCHITECTURES` in
`CMakeLists.txt` for other GPUs.

## Models

| file | size | url |
|------|------|-----|
| `models/ggml-large-v3.bin` | 3.0 GB | [huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin) |
| `models/ggml-silero-v6.2.0.bin` | 0.9 MB | [huggingface.co/ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin) |

The silero VAD model ships in the repo. large-v3 is downloaded separately
(SHA256 `64d182b4…94d1e2`).

## Performance

RTX 3070, large-v3, beam 5, flash attention, VAD:

| content | speed |
|---------|-------|
| dialogue-heavy episode | ~19x realtime |
| full season (26 × 25 min) | ~45 min end-to-end |

## Third party

whisper.cpp + ggml (MIT), FFmpeg (LGPL/GPL build), NVIDIA CUDA runtime
(redistributable), silero-vad (MIT). This project is MIT licensed — see
[LICENSE](LICENSE). The whisper.cpp modification in `patches/` is released
under the same MIT license as whisper.cpp.
