# ===========================================================================
#  resume-benchmarks.ps1 -- the remaining measurement queue
#
#  Every run appends to wiki/benchmarks/results.tsv the moment it finishes, so
#  a crash costs at most the single run in flight.
#
#  ALREADY DONE (see wiki/06-results.md -- do not repeat):
#    * kvtype   q4_0/q4_0 is on the fast FA path. ub 1024 @130k = 2650 t/s.
#    * nopipe   the lean non-pipelined path is reached by raising -ub until the
#               pipelined reserve fails. Reproducible: 2757/2599/2594.
#    * quality  q4_0 and q8_0 both PASS a 108k needle test (needle-test.ps1).
#    * qfirst   q8_0 frontier mapped: 126976 = 2323 t/s, 130048 = 1850 t/s.
#
#  Usage:
#     .\resume-benchmarks.ps1 -Only mtp        # highest value remaining
#     .\resume-benchmarks.ps1                  # everything below
#     .\resume-benchmarks.ps1 -Model <other.gguf> -OutDir <dir>   # override the defaults
# ===========================================================================
param(
  [ValidateSet('all','split','deep','loadmode')]
  [string] $Only   = 'all',
  [string] $Model  = 'S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf',
  [string] $OutDir
)

# Default -OutDir to this repo's own evidence trail, wiki/benchmarks/, located by walking
# up from this script to the folder holding llama-server.exe. Self-locating so moving the
# folder needs no edit here; pass -OutDir to send a run somewhere else.
if (-not $OutDir) {
  $repoRoot = $PSScriptRoot
  while ($repoRoot -and -not (Test-Path (Join-Path $repoRoot 'llama-server.exe'))) {
    $repoRoot = Split-Path $repoRoot -Parent
  }
  if (-not $repoRoot) { throw "Could not locate the repo root above $PSScriptRoot. Pass -OutDir explicitly." }
  $OutDir = Join-Path $repoRoot 'wiki\benchmarks'
}

# bench-harness.ps1 is the single source of truth and lives alongside this script under
# .claude/skills/tuning-llamacpp-configs/scripts/ (see CLAUDE.md). Its portable default
# output dir is bench-results/ next to the binaries; -OutDir here keeps this repo's own
# campaign landing in wiki/benchmarks/ as it always has.
. (Join-Path $PSScriptRoot 'bench-harness.ps1') -Model $Model -OutDir $OutDir

$q8   = @('-ctk','q8_0','-ctv','q8_0')
$q4   = @('-ctk','q4_0','-ctv','q4_0')
$base = @('-fa','on','-ngl','999','-fit','off')

# ---------------------------------------------------------------------------
# SUITE split -- re-map the tensor-split boundary for the WINNING config.
# The -ts 12,29 optimum was found with q8_0 at ub 256, under pipelined
# execution. The best config today is q4_0 at ub 1024 on the lean path, which
# has a completely different memory layout -- so the optimum may have moved,
# and moving it is worth 10-20% prefill.
# ---------------------------------------------------------------------------
if ($Only -in 'all','split') {
  Write-Host "`n=== SUITE split : -ts boundary for q4_0 ub1024 @130k ===" -ForegroundColor Cyan
  foreach ($ts in '10,31','11,30','12,29','13,28','14,27') {
    Probe "q4 ub1024 ts$ts" (@('-ub','1024','-b','2048','-ts',$ts) + $q4 + $base) `
      -Suite 'split-q4' -Ctx 130048 -Npp '8192' | Out-Null
  }
}

# ---------------------------------------------------------------------------
# SUITE deep -- prefill degrades with DEPTH, and depth is what users feel.
# The needle test gave one point at 107,743 tokens; this fills in the curve so
# we can tell a user "a 60k prompt costs you about N seconds".
# NOTE: slow -- a 98k prefill is ~60 s per run.
# ---------------------------------------------------------------------------
if ($Only -in 'all','deep') {
  Write-Host "`n=== SUITE deep : prompt-depth curve for both finalists ===" -ForegroundColor Cyan
  foreach ($npp in 16384, 32768, 65536, 98304) {
    Probe "deep q4 ub1024 npp$npp" (@('-ub','1024','-b','2048','-ts','12,29') + $q4 + $base) `
      -Suite 'deep-q4' -Ctx 130048 -Npp "$npp" -Ntg '128' | Out-Null
    Probe "deep q8 ub512 npp$npp"  (@('-ub','512','-b','2048','-ts','12,29') + $q8 + $base) `
      -Suite 'deep-q8' -Ctx 130048 -Npp "$npp" -Ntg '128' | Out-Null
  }
}

# ---------------------------------------------------------------------------
# SUITE loadmode -- llama.cpp warns that CPU tensor overrides are slower with
# mmap enabled. -ncmoe is a last resort, but if it ever becomes unavoidable,
# does --load-mode none rescue any of the 69% prefill loss?
# ---------------------------------------------------------------------------
if ($Only -in 'all','loadmode') {
  Write-Host "`n=== SUITE loadmode : does --load-mode none rescue -ncmoe? ===" -ForegroundColor Cyan
  Probe "ncmoe2 mmap"     (@('-ub','512','-b','2048','-ts','12,29','-ncmoe','2') + $q8 + $base) `
    -Suite 'loadmode' -Ctx 130048 -Npp '8192' | Out-Null
  Probe "ncmoe2 loadnone" (@('-ub','512','-b','2048','-ts','12,29','-ncmoe','2','--load-mode','none') + $q8 + $base) `
    -Suite 'loadmode' -Ctx 130048 -Npp '8192' | Out-Null
}

Write-Host "`n=== SUMMARY (successful runs, best prefill first) ===" -ForegroundColor Cyan
# Out-String: a bare Format-Table emits format objects that get mangled if the
# caller pipes this script's output anywhere.
Write-Host ((Import-Csv -Delimiter "`t" $Global:LCB_TSV |
  Where-Object { $_.status -eq 'OK' } |
  Sort-Object { [double]$_.pp_ts } -Descending |
  Select-Object -First 20 suite, label, ctx, npp, pp_ts, tg_ts, vram_total |
  Format-Table -AutoSize | Out-String).TrimEnd())

Write-Host "Full data: $Global:LCB_TSV"
Write-Host "Speculative decoding cannot be measured here -- it needs llama-server." -ForegroundColor DarkGray
Write-Host "Use: .\mtp-test.ps1 -Model <model.gguf>   (see wiki/09-speculative-decoding.md)" -ForegroundColor DarkGray
