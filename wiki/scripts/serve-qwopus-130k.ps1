# ===========================================================================
#  Profile A -- Qwopus Q4_K_M, full 130k context, q8_0 KV (QUALITY-FIRST)
#
#  Measured: 1850 t/s prefill synthetic
#            1414 t/s prefill and 47.5 t/s generation on a REAL 107,743-token
#            request over HTTP. Needle test: PASS.
#
#  Use this when quality matters more than latency and you need the full
#  130k window. For 43% more prefill at the cost of a lossier KV cache, see
#  serve-qwopus-q4-130k.ps1. For the best all-round config, see
#  serve-qwopus-fast.ps1 (127k, same q8_0 KV, 2323 t/s).
#
#  NOTE: this config sits within a few hundred MiB of the VRAM ceiling. If it
#  fails to load, close GPU-using desktop apps and check:
#      nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
# ===========================================================================
param(
  [int]    $Port   = 9010,
  [int]    $Ctx    = 130048,
  [int]    $Ubatch = 512,
  [switch] $Background
)

$ErrorActionPreference = 'Stop'
$LC    = 'S:\OneDrive\Tools\llamacpp'
$Model = 'S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf'

if (-not (Test-Path $Model)) { throw "Model not found: $Model" }

# Deliberately NOT set: $env:GGML_SCHED_MAX_COPIES.
# It is a COMPILE-TIME #define, not a runtime variable -- setting it does
# nothing. To disable pipeline parallelism deterministically, rebuild with
#     cmake -B build -DGGML_CUDA=ON -DLLAMA_SCHED_MAX_COPIES=1
# See wiki/06-results.md 6.2.

$argv = @(
  '-m', $Model
  '--alias', 'qwopus-coder'
  '--host', '0.0.0.0'
  '--port', "$Port"

  # --- context ---
  '-c', "$Ctx"
  '-np', '1'                 # ONE slot: -np N divides the context by N
  '-ctk', 'q8_0'             # symmetric pair on the compiled flash-attn path
  '-ctv', 'q8_0'             # NEVER mix KV types: costs ~89% of prefill

  # --- placement ---
  '-ngl', '999'              # every layer on GPU; no CPU expert offload
  '-sm',  'layer'
  '-ts',  '12,29'            # the only ratio that loads at 130k
  '-fa',  'on'               # mandatory: -fa off costs 9 GiB of buffers

  # --- batching ---
  '-b',  '2048'
  '-ub', "$Ubatch"           # 512; ub 1024 does not fit with q8_0 at 130k

  '-fit', 'off'              # do not let the auto-fitter alter a measured config

  # --- prompt cache: the single biggest real-world win (34x on repeat prefill)
  '-cram', '24576'

  # --- sampling: model authors' "precise coding" preset ---
  '--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--min-p', '0.0'
)

Write-Host "Profile A (quality-first): ctx=$Ctx ub=$Ubatch KV=q8_0/q8_0 port=$Port" -ForegroundColor Cyan

if ($Background) {
  $log = Join-Path $LC 'wiki\benchmarks\server.log'
  Start-Process -FilePath (Join-Path $LC 'llama-server.exe') -ArgumentList $argv `
    -WindowStyle Hidden -RedirectStandardError $log
  Write-Host "Running in background. Log: $log"
  Write-Host "Stop with: Get-Process llama-server | Stop-Process"
} else {
  & (Join-Path $LC 'llama-server.exe') @argv
}
