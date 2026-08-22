# ============================================================================
#  bench-harness.ps1  --  reusable llama.cpp benchmark harness
# ----------------------------------------------------------------------------
#  Why this exists:
#    Tuning llama.cpp on a memory-constrained multi-GPU box means running the
#    same binary dozens of times with slightly different flags. Doing that by
#    hand is slow and, worse, easy to get wrong: a config that *looks* fast can
#    actually be spilling into system RAM, and a crash loses everything you
#    measured so far.
#
#  What it guarantees:
#    1. FRESH PROCESS per config. VRAM fragmentation from a previous run must
#       never contaminate the next measurement.
#    2. DURABLE, APPEND-ONLY results. Every run is flushed to results.tsv the
#       moment it finishes, so a driver crash or OOM reboot costs you at most
#       the single run that was in flight.
#    3. PEAK VRAM per GPU, sampled independently of llama.cpp, so you can see
#       whether a config really fit or quietly overflowed.
#
#  Dot-source this file, then call Probe.
# ============================================================================

$Global:LC_DIR   = "S:\OneDrive\Tools\llamacpp"
$Global:BENCHDIR = Join-Path $LC_DIR "wiki\benchmarks"
$Global:LOGDIR   = Join-Path $BENCHDIR "logs"
$Global:RESULTS  = Join-Path $BENCHDIR "results.tsv"

New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

# Create the results file with a header exactly once.
if (-not (Test-Path $RESULTS)) {
  "run_utc`tsuite`tlabel`tstatus`tmodel`tctx`tnpp`tntg`tpp_ts`ttg_ts`tvram0`tvram1`tvram_total`tsecs`targs" |
    Out-File -Encoding utf8 $RESULTS
}

# ---------------------------------------------------------------------------
# Start-VramSampler / Stop-VramSampler
#   nvidia-smi's own "-f file" logging buffers on Windows and often yields an
#   empty file for short runs, so we poll it ourselves from a background job
#   and keep the maximum we ever saw.
# ---------------------------------------------------------------------------
function Start-VramSampler {
  param([string]$Path)
  if (Test-Path $Path) { Remove-Item $Path -Force }
  Start-Job -ScriptBlock {
    param($p)
    while ($true) {
      try {
        $line = (nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits) -join ';'
        Add-Content -Path $p -Value $line -ErrorAction SilentlyContinue
      } catch {}
      Start-Sleep -Milliseconds 350
    }
  } -ArgumentList $Path
}

function Stop-VramSampler {
  param($Job, [string]$Path)
  try { Stop-Job  $Job -ErrorAction SilentlyContinue } catch {}
  try { Remove-Job $Job -Force -ErrorAction SilentlyContinue } catch {}
  $p0 = 0; $p1 = 0
  if (Test-Path $Path) {
    foreach ($line in Get-Content $Path) {
      foreach ($entry in ($line -split ';')) {
        $q = $entry -split ',\s*'
        if ($q.Count -ge 2) {
          $i = 0; $v = 0
          if ([int]::TryParse($q[0].Trim(), [ref]$i) -and [int]::TryParse($q[1].Trim(), [ref]$v)) {
            if ($i -eq 0 -and $v -gt $p0) { $p0 = $v }
            if ($i -eq 1 -and $v -gt $p1) { $p1 = $v }
          }
        }
      }
    }
  }
  return @($p0, $p1)
}

# ---------------------------------------------------------------------------
# Probe -- run ONE configuration and record it.
#   -Label   short human name, also used for the per-run log filename
#   -A       array of extra llama.cpp flags under test
#   -Suite   grouping tag written into results.tsv
# ---------------------------------------------------------------------------
function Probe {
  param(
    [string]   $Label,
    [string[]] $A,
    [string]   $Suite = "misc",
    [string]   $Model = "S:\HuggingFace\lmstudio\Jackrong\Qwopus3.6-35B-A3B-Coder-MTP-GGUF\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf",
    [int]      $Ctx   = 130048,
    [string]   $Npp   = "8192",
    [string]   $Ntg   = "128",
    [int]      $TimeoutSec = 900
  )

  $exe     = Join-Path $LC_DIR "llama-batched-bench.exe"
  $slug    = ($Label -replace '[^\w\.\-]', '_')
  $logFile = Join-Path $LOGDIR "$slug.txt"
  $smiFile = Join-Path $env:TEMP "vram_$slug.csv"

  $sampler = Start-VramSampler -Path $smiFile
  $sw = [Diagnostics.Stopwatch]::StartNew()

  $argv = @("-m", $Model, "-c", "$Ctx", "-npp", $Npp, "-ntg", $Ntg, "-npl", "1") + $A
  $out  = & $exe @argv 2>&1 | ForEach-Object { [string]$_ }

  $secs = [math]::Round($sw.Elapsed.TotalSeconds, 1)
  $peak = Stop-VramSampler -Job $sampler -Path $smiFile

  # Persist the FULL log first -- if parsing throws, the evidence survives.
  $out | Out-File -Encoding utf8 $logFile

  $pp = ""; $tg = ""; $status = "OK"
  foreach ($l in $out) {
    if ($l -match '^\|\s*\d+\s*\|') {
      $c = ($l -split '\|') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
      if ($c.Count -ge 8) { $pp = $c[5]; $tg = $c[7] }
    }
    if ($l -match 'out of memory')        { $status = "OOM" }
    elseif ($l -match 'failed to (load|create)') { if ($status -eq "OK") { $status = "FAIL" } }
  }
  # A run that produced a result row SUCCEEDED, even if an allocation failed
  # earlier and llama.cpp retried without pipeline parallelism. Marking those
  # OOM makes consumers discard valid data -- 17 rows were mislabelled that way
  # before this was caught by an external review of results.tsv.
  if ($pp -ne "" -and $status -eq "OOM") { $status = "OK" }
  if ($pp -eq "" -and $status -eq "OK") { $status = "FAIL" }

  # Append-only: flush this single run to disk immediately.
  $row = @(
    (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"),
    $Suite, $Label, $status, (Split-Path $Model -Leaf), $Ctx, $Npp, $Ntg,
    $pp, $tg, $peak[0], $peak[1], ($peak[0] + $peak[1]), $secs, ($A -join ' ')
  ) -join "`t"
  Add-Content -Path $RESULTS -Value $row

  Write-Host ("{0,-32} {1,-4} PP={2,9} TG={3,8}  VRAM {4,5}/{5,5}={6,5} MiB  {7,6}s" -f `
    $Label, $status, $pp, $tg, $peak[0], $peak[1], ($peak[0] + $peak[1]), $secs)

  return [pscustomobject]@{
    Label = $Label; Status = $status; PP = $pp; TG = $tg
    V0 = $peak[0]; V1 = $peak[1]; VT = ($peak[0] + $peak[1]); Secs = $secs; Args = ($A -join ' ')
  }
}
