# ===========================================================================
#  Profile B -- Qwopus Q4_K_M, full 130k context, q4_0 KV (SPEED-FIRST)
#
#  Measured: 2650 t/s prefill synthetic (2757 / 2599 / 2594 over three runs)
#            1758 t/s prefill and 46.6 t/s generation on a REAL 107,743-token
#            request -- 15 seconds faster end-to-end than Profile A.
#            Needle test at 108k tokens: PASS (retrieved CRIMSON-PELICAN-4417).
#
#  WHY IT IS FAST -- two effects that compound:
#    1. q4_0 KV is ~700 MiB smaller than q8_0, which is exactly what makes
#       -ub 1024 reachable at 130k.
#    2. -ub 1024's pipelined compute buffers do NOT fit, so llama.cpp retries
#       without pipeline parallelism -- and on this mismatched 3070/5060 Ti
#       pair that lean path is substantially faster (2650 vs 1850 t/s).
#
#  EXPECT AN ALARMING LOG LINE. You will see:
#      graph_reserve: failed to allocate compute buffers
#      sched_reserve: compute buffer allocation failed, retrying without
#                     pipeline parallelism
#  That is this profile WORKING AS INTENDED. Confirm "model loaded" and
#  "listening on" follow it.
#
#  QUALITY CAVEAT: q4_0 KV passed a 108k needle-RETRIEVAL test. That does not
#  prove it is lossless for multi-step reasoning or code-edit fidelity. If
#  quality is paramount, use serve-qwopus-130k.ps1 (q8_0) instead.
# ===========================================================================
param(
  [int]    $Port   = 9010,
  [int]    $Ctx    = 130048,
  [int]    $Ubatch = 1024,
  [switch] $Background
)

$ErrorActionPreference = 'Stop'
# Repo root = the folder holding llama-server.exe, found by walking up from this
# script under .claude/skills/tuning-llamacpp-configs/scripts/. Self-locating (the
# same idiom bench-harness.ps1 uses) so moving the folder needs no edit here.
$LC = $PSScriptRoot
while ($LC -and -not (Test-Path (Join-Path $LC 'llama-server.exe'))) { $LC = Split-Path $LC -Parent }
if (-not $LC) { throw "llama-server.exe not found in any parent of $PSScriptRoot" }
$Model = 'S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf'

if (-not (Test-Path $Model)) { throw "Model not found: $Model" }

$argv = @(
  '-m', $Model
  '--alias', 'qwopus-coder'
  '--host', '0.0.0.0', '--port', "$Port"

  '-c', "$Ctx", '-np', '1'
  '-ctk', 'q4_0', '-ctv', 'q4_0'   # SYMMETRIC: q4_0/q4_0 has a compiled FA kernel.
                                   # q8_0/q4_0 would NOT and costs 89% of prefill.

  '-ngl', '999', '-sm', 'layer', '-ts', '12,29', '-fa', 'on'
  '-b', '2048', '-ub', "$Ubatch"
  '-fit', 'off'
  '-cram', '24576'
  '--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--min-p', '0.0'
)

Write-Host "Profile B (speed-first): ctx=$Ctx ub=$Ubatch KV=q4_0/q4_0 port=$Port" -ForegroundColor Cyan
Write-Host "A 'retrying without pipeline parallelism' warning is EXPECTED here." -ForegroundColor DarkGray

if ($Background) {
  $log = Join-Path $LC 'wiki\benchmarks\server.log'
  Start-Process -FilePath (Join-Path $LC 'llama-server.exe') -WorkingDirectory $LC -ArgumentList $argv `
    -WindowStyle Hidden -RedirectStandardError $log
  Write-Host "Running in background. Log: $log"
} else {
  Push-Location $LC
  try { & (Join-Path $LC 'llama-server.exe') @argv } finally { Pop-Location }
}
