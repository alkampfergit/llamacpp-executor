# ===========================================================================
#  Profile C -- Qwen3.6-35B-A3B UD-Q3_K_XL at 130k context
#
#  The comfortable option. At 15.69 GiB the model leaves ~2.5 GiB spare, so
#  you can keep f16 KV cache AND -ub 512 -- no compromises needed.
#
#  Trade-off: Q3 quality. THAT IS ALL.
#
#  This header used to claim "~20% slower GENERATION than the Q4_K_M model
#  (94 t/s vs 117 t/s) because Q3_K CUDA kernels are slower than Q4_K ones."
#  That comparison crossed a fine-tune boundary -- 94 was this file, 117 was
#  Jackrong's Qwopus, a different model from a different publisher.
#
#  Measured properly (3 quants of the SAME base model, identical config, 3 reps,
#  wiki chapter 16.3): generation is 90.5 / 91.5 / 91.3 t/s -- a 1.0% spread, so
#  quant-independent. And this file has the FASTEST PREFILL of the three:
#  2778 t/s vs 2429 (UD-Q4_K_S) and 1579 (lmstudio Q4_K_M), at ~3.8 GiB less
#  VRAM. On this box the smaller quant is strictly faster.
# ===========================================================================
param(
  [int]    $Port = 9010,
  [int]    $Ctx  = 130048,
  [string] $Split = '1,2',
  [switch] $Background
)

$ErrorActionPreference = 'Stop'
# Repo root = the folder holding llama-server.exe, found by walking up from this
# script under .claude/skills/tuning-llamacpp-configs/scripts/. Self-locating (the
# same idiom bench-harness.ps1 uses) so moving the folder needs no edit here.
$LC = $PSScriptRoot
while ($LC -and -not (Test-Path (Join-Path $LC 'llama-server.exe'))) { $LC = Split-Path $LC -Parent }
if (-not $LC) { throw "llama-server.exe not found in any parent of $PSScriptRoot" }
$Model = 'S:\HuggingFace\lmstudio\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf'

if (-not (Test-Path $Model)) { throw "Model not found: $Model" }
$argv = @(
  '-m', $Model
  '--alias', 'qwen3.6-35b'
  '--host', '0.0.0.0', '--port', "$Port"
  '-c', "$Ctx", '-np', '1'

  # NO -ctk/-ctv here: f16 KV fits, and f16 is the fastest KV there is.
  # Only quantise the KV cache when you actually have to.

  '-ngl', '999', '-sm', 'layer', '-ts', $Split, '-fa', 'on'
  '-b', '2048', '-ub', '512'
  '-fit', 'off'
  '-cram', '24576'
  '--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--min-p', '0.0'
)

# Deliberately NOT passed, and why:
#   -mg 1            does nothing with -sm layer
#   --cache-prompt   accepted but a no-op in build 10509 (now the default)
#   -kvo             same
#   -ncmoe           would cost 44% of prefill for 2 layers

Write-Host "Starting llama-server (Qwen3.6 Q3_K_XL) on port $Port  (ctx=$Ctx, ts=$Split)" -ForegroundColor Cyan
Write-Host "Tip: -ts 1,3 also works; 1,2 relieves the display GPU. Try both." -ForegroundColor DarkGray

if ($Background) {
  $log = Join-Path $LC 'wiki\benchmarks\server.log'
  Start-Process -FilePath (Join-Path $LC 'llama-server.exe') -WorkingDirectory $LC -ArgumentList $argv `
    -WindowStyle Hidden -RedirectStandardError $log
  Write-Host "Running in background. Log: $log"
} else {
  Push-Location $LC
  try { & (Join-Path $LC 'llama-server.exe') @argv } finally { Pop-Location }
}

