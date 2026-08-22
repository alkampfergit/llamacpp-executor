# ===========================================================================
#  Profile C -- Qwopus Q4_K_M at 127k with q8_0 KV (BEST ALL-ROUND)
#
#  Measured: 2323 t/s prefill, 109 t/s generation -- the fastest q8_0
#            configuration found, and faster than Profile B's q4_0 at ub 512.
#
#  THE TRADE: giving up 3072 tokens of context (2.4% of the window) buys 26%
#  more prefill, while KEEPING the higher-precision q8_0 KV cache.
#
#      -c 130048  ->  1850 t/s
#      -c 129024  ->  2101 t/s
#      -c 126976  ->  2323 t/s   <- here
#      -c 122880  ->  2225 t/s
#
#  The curve is steep right at the ceiling because that is where VRAM pressure
#  starts forcing compromises. If you can live with 127k instead of 130k, this
#  is the best trade in the whole wiki.
# ===========================================================================
param(
  [int]    $Port   = 9010,
  [int]    $Ctx    = 126976,
  [int]    $Ubatch = 512,
  [switch] $Background
)

$ErrorActionPreference = 'Stop'
$LC    = 'S:\OneDrive\Tools\llamacpp'
$Model = 'S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf'

if (-not (Test-Path $Model)) { throw "Model not found: $Model" }

$argv = @(
  '-m', $Model
  '--alias', 'qwopus-coder'
  '--host', '0.0.0.0', '--port', "$Port"
  '-c', "$Ctx", '-np', '1'
  '-ctk', 'q8_0', '-ctv', 'q8_0'
  '-ngl', '999', '-sm', 'layer', '-ts', '12,29', '-fa', 'on'
  '-b', '2048', '-ub', "$Ubatch"
  '-fit', 'off'
  '-cram', '24576'
  '--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--min-p', '0.0'
)

Write-Host "Profile C (best all-round): ctx=$Ctx ub=$Ubatch KV=q8_0/q8_0 port=$Port" -ForegroundColor Cyan

if ($Background) {
  $log = Join-Path $LC 'wiki\benchmarks\server.log'
  Start-Process -FilePath (Join-Path $LC 'llama-server.exe') -WorkingDirectory $LC -ArgumentList $argv `
    -WindowStyle Hidden -RedirectStandardError $log
  Write-Host "Running in background. Log: $log"
} else {
  Push-Location $LC
  try { & (Join-Path $LC 'llama-server.exe') @argv } finally { Pop-Location }
}
