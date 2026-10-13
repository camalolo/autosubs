# make_release.ps1 - build a portable Windows release of autosubs
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File make_release.ps1
#   powershell -ExecutionPolicy Bypass -File make_release.ps1 -WithModels        # also bundle large-v3 (2.9 GB)
#   powershell -ExecutionPolicy Bypass -File make_release.ps1 -NoFfmpeg -NoZip   # exe+dlls only, no archive
#
# Produces:
#   release\autosubs-<version>-win64-cuda\        (portable folder)
#   release\autosubs-<version>-win64-cuda.zip     (unless -NoZip)

param(
    [switch]$NoFfmpeg,
    [switch]$WithModels,
    [switch]$NoZip
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# ---------------------------------------------------------------- build
Write-Host "== building ==" -ForegroundColor Cyan
$buildLog = Join-Path $env:TEMP 'autosubs_release_build.log'
& cmd /c "`"$root\build_autosubs.bat`" > `"$buildLog`" 2>&1"
if (-not (Select-String -Path $buildLog -Pattern 'BUILD_OK' -Quiet)) {
    Get-Content $buildLog -Tail 30
    throw "build failed (see $buildLog)"
}

# ---------------------------------------------------------------- version
$ver = '1.0.0'
$help = & (Join-Path $root 'bin\autosubs.exe') --help 2>&1 | Select-Object -First 1
if ($help -match 'autosubs\s+(\S+)\s') { $ver = $Matches[1] }
Write-Host "== version: $ver ==" -ForegroundColor Cyan

$name  = "autosubs-$ver-win64-cuda"
$rel   = Join-Path $root 'release'
$stage = Join-Path $rel $name
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage 'models') | Out-Null

# ---------------------------------------------------------------- binaries
Write-Host "== staging binaries ==" -ForegroundColor Cyan
$bin = Join-Path $root 'bin'
$dlls = @(
    'whisper.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll', 'ggml-cuda.dll',
    'cublas64_13.dll', 'cublasLt64_13.dll'
)
Copy-Item (Join-Path $bin 'autosubs.exe') $stage
foreach ($d in $dlls) {
    $src = Join-Path $bin $d
    if (-not (Test-Path $src)) { throw "missing $d in bin\ - run build_autosubs.bat first" }
    Copy-Item $src $stage
}

# MSVC runtime (app-local deployment is permitted with the VC++ toolset)
foreach ($d in @('VCRUNTIME140.dll', 'VCRUNTIME140_1.dll', 'MSVCP140.dll')) {
    $src = Join-Path $env:SystemRoot "System32\$d"
    if (Test-Path $src) { Copy-Item $src $stage } else { Write-Warning "MSVC runtime $d not found" }
}

# ---------------------------------------------------------------- ffmpeg
if (-not $NoFfmpeg) {
    Write-Host "== staging ffmpeg ==" -ForegroundColor Cyan
    foreach ($tool in @('ffmpeg.exe', 'ffprobe.exe')) {
        $cmd = Get-Command $tool -ErrorAction SilentlyContinue
        if ($cmd) { Copy-Item $cmd.Source $stage } else { Write-Warning "$tool not found in PATH - release will need ffmpeg installed" }
    }
}

# ---------------------------------------------------------------- models
Write-Host "== staging models ==" -ForegroundColor Cyan
$silero = Join-Path $root 'models\ggml-silero-v6.2.0.bin'
if (Test-Path $silero) { Copy-Item $silero (Join-Path $stage 'models') }

$large = Join-Path $root 'models\ggml-large-v3.bin'
if ($WithModels) {
    if (Test-Path $large) { Copy-Item $large (Join-Path $stage 'models') }
    else { Write-Warning "models\ggml-large-v3.bin not found - not bundled" }
}

# ---------------------------------------------------------------- readme
$modelBlock = if ($WithModels) {
    "  models\ggml-large-v3.bin   (bundled)"
} else {
    "  models\ggml-large-v3.bin   NOT bundled - download it (see below)"
}
@"
autosubs $ver - automatic, GPU-accelerated SRT generation (whisper.cpp large-v3)
================================================================================

QUICK START
-----------
  autosubs.exe "E:\path\to\video.mkv"
  autosubs.exe "E:\path\to\series folder"

A .srt file is written next to every video. Existing .srt files are skipped
(use --force to regenerate). Run "autosubs.exe --help" for all options.

CONTENTS
--------
  autosubs.exe               the tool
  whisper.dll ggml*.dll      whisper.cpp build (CUDA)
  cublas*.dll                NVIDIA CUDA runtime libraries
  ffmpeg.exe ffprobe.exe     audio demuxing/decoding
  models\ggml-silero-v6.2.0.bin   voice-activity detection (fast, recommended)
$modelBlock

REQUIREMENTS
------------
  * 64-bit Windows 10/11
  * NVIDIA GPU with compute capability 8.6 (this build targets sm_86:
    RTX 30xx, A10/A40, and other Ampere parts) with a recent driver.
    On other GPUs rebuild with -DCMAKE_CUDA_ARCHITECTURES set for your card,
    or run with --no-gpu (much slower).
  * ~4 GB free VRAM for large-v3, ~6 GB disk for the model.

MODEL DOWNLOAD (if not bundled)
-------------------------------
  Put the file in the models\ folder next to autosubs.exe:

    models\ggml-large-v3.bin   (3.0 GB)
      https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin

  Verify (SHA256):
    64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2

SPEED / QUALITY
---------------
  Defaults: large-v3, beam search 5, flash attention, GPU, silero VAD.
  On an RTX 3070 expect roughly 15-20x realtime (a 25 min episode ~1.5 min).

  Options worth knowing:
    -l ja              force language
    --translate        translate to English
    --prompt "text"    bias names/jargon
    --beam 1           greedy decode (faster, slightly worse)
    --denoise model.rnnn  RNNoise (opt-in; can harm music-heavy audio)
    --no-vad           disable voice-activity detection
    --no-gpu           CPU only

LICENSE / THIRD PARTY
---------------------
  whisper.cpp (MIT), ggml (MIT), FFmpeg (LGPL/GPL as built by gyan.dev),
  NVIDIA CUDA runtime (NVIDIA EULA, redistributable).
"@ | Set-Content -Path (Join-Path $stage 'README.txt') -Encoding UTF8

# ---------------------------------------------------------------- zip
$size = (Get-ChildItem $stage -Recurse | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("== staged {0} ({1:N0} MB) ==" -f $stage, $size) -ForegroundColor Cyan

if (-not $NoZip) {
    $zip = Join-Path $rel "$name.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Write-Host "== zipping ==" -ForegroundColor Cyan
    Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
    $zsize = (Get-Item $zip).Length / 1MB
    Write-Host ("== done: {0} ({1:N0} MB) ==" -f $zip, $zsize) -ForegroundColor Green
} else {
    Write-Host "== done (no zip) ==" -ForegroundColor Green
}
