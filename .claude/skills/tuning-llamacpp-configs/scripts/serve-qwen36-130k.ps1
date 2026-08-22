# ===========================================================================
#  Profile C -- Qwen3.6-35B-A3B UD-Q3_K_XL at 130k context
#
#  The comfortable option. At 15.69 GiB the model leaves ~2.5 GiB spare, so
#  you can keep f16 KV cache AND -ub 512 -- no compromises needed.
#
#  Trade-off: Q3 quality, and ~20% slower GENERATION than the Q4_K_M model
#  (94 t/s vs 117 t/s) because Q3_K CUDA kernels are slower than Q4_K ones.
#  A smaller file is not automatically a faster model.
# ===========================================================================
param(
  [int]    $Port = 9010,
  [int]    $Ctx  = 130048,
  [string] $Split = '1,2',
  [switch] $Background
)

$ErrorActionPreference = 'Stop'
$LC    = 'S:\OneDrive\Tools\llamacpp'
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

