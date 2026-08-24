# ===========================================================================
#  serve-qwen38-27b.ps1 -- the four measured profiles for Qwen3.8-27B Q4_K_M
#
#  See wiki/10-results-qwen38-27b.md for the numbers behind each one.
#
#    A  130k, q8_0 KV, draft-mtp n4   47 t/s shallow / 21 t/s @108k, peak 23560
#                                     <- daily driver, but NO display-GPU margin
#    B  130k, q4_0 KV, draft-mtp n4   47 t/s shallow / 26 t/s @108k, peak 22893
#                                     <- same speed, 670 MiB more room
#    C  130k, q8_0 KV, no drafting    22 t/s shallow / 15 t/s @108k, peak 22453
#                                     <- biggest margin, fastest prefill (830 t/s)
#    D  192k, q4_0 KV, no drafting    22 t/s shallow, peak 22126
#                                     <- only q4_0 reaches past 130k. Quality
#                                        beyond 108k is UNVALIDATED.
#
#  Usage:  ./serve-qwen38-27b.ps1                 # profile A
#          ./serve-qwen38-27b.ps1 -Profile C -Background
# ===========================================================================
param(
  [ValidateSet('A','B','C','D')] [string] $Profile = 'A',
  [int]    $Port  = 9020,
  [string] $Split = '22,43',      # measured optimum; 23,42 FAILS (3070 runs out)
  [int]    $Ctx   = 0,            # 0 = the profile's own value
  [switch] $Background
)

$ErrorActionPreference = 'Stop'
# Repo root = the folder holding llama-server.exe, found by walking up from this
# script under .claude/skills/tuning-llamacpp-configs/scripts/. Self-locating (the
# same idiom bench-harness.ps1 uses) so moving the folder needs no edit here.
$LC = $PSScriptRoot
while ($LC -and -not (Test-Path (Join-Path $LC 'llama-server.exe'))) { $LC = Split-Path $LC -Parent }
if (-not $LC) { throw "llama-server.exe not found in any parent of $PSScriptRoot" }
$Model = 'S:\HuggingFace\lmstudio\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf'
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }

$cfg = switch ($Profile) {
  'A' { @{ Ctx = 130048; Kv = 'q8_0'; Mtp = $true  } }
  'B' { @{ Ctx = 130048; Kv = 'q4_0'; Mtp = $true  } }
  'C' { @{ Ctx = 130048; Kv = 'q8_0'; Mtp = $false } }
  'D' { @{ Ctx = 196608; Kv = 'q4_0'; Mtp = $false } }
}
if ($Ctx -gt 0) { $cfg.Ctx = $Ctx }

$argv = @(
  '-m', $Model
  '--alias', 'qwen3.8-27b'
  '--host', '127.0.0.1', '--port', "$Port"

  # -np 1 is load-bearing: with -np N each slot gets only -c / N tokens.
  '-c', "$($cfg.Ctx)", '-np', '1'

  # KV type: ONLY a symmetric pair from the compiled FA set (f16/bf16/q8_0/q4_0).
  # f16 is NOT an option at this context -- 16 attention layers x n_embd_k_gqa
  # 1024 means 64 MiB of f16 KV per 1024 tokens, so 130k would need 8128 MiB.
  '-ctk', $cfg.Kv, '-ctv', $cfg.Kv

  '-ngl', '999', '-sm', 'layer', '-ts', $Split, '-fa', 'on'

  # -ub 512 measured flat from 256 to 2048 on this dense model; 512 is the
  # cheapest point that is not -ub 128 (which costs 22% of prefill).
  '-b', '2048', '-ub', '512'
  '-fit', 'off'

  # Sampling: NOT measured here, adjust to taste.
  '--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--min-p', '0.0'
)

if ($cfg.Mtp) {
  # The MTP head is blk.64. Without this flag the log says "unused tensor
  # blk.64.* -- ignoring" and the head costs nothing; with it, the head costs
  # ~1.1 GiB (a SECOND graph reservation for the draft context) and doubles
  # generation.
  #
  # n-max 4 is a SAFE value on the plateau, not a proven optimum. The sweep it
  # came from was one run per point at temperature 0.6, where run-to-run spread
  # is +-20% because draft acceptance is resampled every run (wiki chapter 15
  # measured r^2 = 0.872 between acceptance and throughput). n-max 3..8 are
  # indistinguishable at that precision. Keeping 4 deliberately: re-tuning off
  # noisy data would be the same mistake in the other direction. If you need the
  # real optimum, sweep with mtp-test.ps1 -Temperature 0.
  $argv += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', '4')
}

# Deliberately NOT passed, and why:
#   -mg               does nothing with -sm layer
#   -ncmoe            meaningless: n_expert = 0, this model is dense
#   -ctkd / -ctvd     does not shrink the MTP overhead (measured: it grew)
#   --spec-type ngram-simple   measured at exactly 0% on this model
#   --jinja           add it if you need tool calling; not used for any measurement

Write-Host ("Qwen3.8-27B profile {0}: ctx={1} kv={2} mtp={3} ts={4} port={5}" -f `
  $Profile, $cfg.Ctx, $cfg.Kv, $cfg.Mtp, $Split, $Port) -ForegroundColor Cyan
if ($Profile -eq 'A') {
  Write-Host "NOTE: profile A peaks at ~23.5 GiB and leaves the display GPU ~400 MiB." -ForegroundColor Yellow
  Write-Host "      If anything else wants VRAM on device 1, use -Profile B instead." -ForegroundColor Yellow
}

if ($Background) {
  $log = Join-Path $LC 'wiki\benchmarks\server.log'
  Start-Process -FilePath (Join-Path $LC 'llama-server.exe') -WorkingDirectory $LC -ArgumentList $argv `
    -WindowStyle Hidden -RedirectStandardError $log
  Write-Host "Running in background. Log: $log"
} else {
  Push-Location $LC
  try { & (Join-Path $LC 'llama-server.exe') @argv } finally { Pop-Location }
}
