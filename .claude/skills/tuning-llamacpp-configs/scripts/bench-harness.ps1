# ===========================================================================
#  bench-harness.ps1 -- portable llama.cpp benchmark harness
#
#  Dot-source it, then call Probe once per configuration:
#
#     . ./bench-harness.ps1 -Model S:/models/foo.gguf
#     Probe "ub512" @('-ub','512','-ts','12,29','-ctk','q8_0','-ctv','q8_0',
#                     '-fa','on','-ngl','999','-fit','off') -Ctx 130048
#
#  It enforces the five rules that make llama.cpp measurements trustworthy:
#
#   1. FRESH PROCESS per configuration. VRAM fragmentation from a previous run
#      makes a later config read up to 40% slower than the identical config
#      measured first.
#   2. llama-batched-bench, not llama-bench -- only it accepts -c, so only it
#      allocates the KV cache the server will actually use.
#   3. PEAK VRAM recorded next to every throughput number. Throughput alone
#      cannot distinguish "fast" from "silently overflowing into system RAM".
#   4. APPEND-ONLY results, flushed before the next run starts. Configurations
#      near the VRAM ceiling crash; a crash must cost one run, not the campaign.
#   5. FULL LOG saved per run, written before parsing, so a silent fallback
#      ("retrying without pipeline parallelism") stays discoverable.
# ===========================================================================
[CmdletBinding()]
param(
  [string] $Model,
  [string] $LlamaDir,
  [string] $OutDir
)

function Find-LlamaDir {
  param([string] $Explicit)
  if ($Explicit) {
    if (Test-Path (Join-Path $Explicit 'llama-batched-bench.exe')) { return $Explicit }
    throw "llama-batched-bench.exe not found in -LlamaDir '$Explicit'"
  }
  $dir = $PSScriptRoot
  for ($i = 0; $i -lt 6 -and $dir; $i++) {
    if (Test-Path (Join-Path $dir 'llama-batched-bench.exe')) { return $dir }
    $dir = Split-Path $dir -Parent
  }
  $onPath = Get-Command 'llama-batched-bench.exe' -ErrorAction SilentlyContinue
  if ($onPath) { return Split-Path $onPath.Source -Parent }
  throw "Could not locate llama-batched-bench.exe. Pass -LlamaDir explicitly."
}

$Global:LCB_DIR   = Find-LlamaDir -Explicit $LlamaDir
$Global:LCB_MODEL = $Model
$Global:LCB_OUT   = if ($OutDir) { $OutDir } else { Join-Path $LCB_DIR 'bench-results' }
$Global:LCB_LOGS  = Join-Path $LCB_OUT 'logs'
$Global:LCB_TSV   = Join-Path $LCB_OUT 'results.tsv'

New-Item -ItemType Directory -Force -Path $LCB_LOGS | Out-Null
if (-not (Test-Path $LCB_TSV)) {
  "run_utc`tsuite`tlabel`tstatus`tmodel`tctx`tnpp`tntg`tpp_ts`ttg_ts`tvram0`tvram1`tvram_total`tsecs`targs" |
    Out-File -Encoding utf8 $LCB_TSV
}

# nvidia-smi's own "-f file" logging buffers on Windows and often produces an
# empty file for short runs, so poll it from a background job instead.
function Start-VramSampler {
  param([string] $Path)
  if (Test-Path $Path) { Remove-Item $Path -Force }
  Start-Job -ScriptBlock {
    param($p)
    while ($true) {
      try {
        Add-Content -Path $p -ErrorAction SilentlyContinue -Value `
          ((nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits) -join ';')
      } catch {}
      Start-Sleep -Milliseconds 350
    }
  } -ArgumentList $Path
}

function Stop-VramSampler {
  param($Job, [string] $Path)
  try { Stop-Job  $Job -ErrorAction SilentlyContinue } catch {}
  try { Remove-Job $Job -Force -ErrorAction SilentlyContinue } catch {}
  $peak = @{}
  if (Test-Path $Path) {
    foreach ($line in Get-Content $Path) {
      foreach ($entry in ($line -split ';')) {
        $q = $entry -split ',\s*'
        if ($q.Count -ge 2) {
          $i = 0; $v = 0
          if ([int]::TryParse($q[0].Trim(), [ref]$i) -and [int]::TryParse($q[1].Trim(), [ref]$v)) {
            if (-not $peak.ContainsKey($i) -or $v -gt $peak[$i]) { $peak[$i] = $v }
          }
        }
      }
    }
  }
  return $peak
}

<#
.SYNOPSIS
  Run ONE llama.cpp configuration and record it durably.
.PARAMETER Label
  Short name; also the per-run log filename.
.PARAMETER A
  Extra llama.cpp flags under test. Always include -fa on and -fit off.
.PARAMETER Ctx
  Real context size. Must match what you intend to serve, or the KV cache
  allocation -- and therefore the whole memory picture -- is wrong.
#>
function Probe {
  param(
    [Parameter(Mandatory = $true)] [string]   $Label,
    [Parameter(Mandatory = $true)] [string[]] $A,
    [string] $Suite = 'misc',
    [string] $Model = $Global:LCB_MODEL,
    [int]    $Ctx   = 32768,
    [string] $Npp   = '8192',
    [string] $Ntg   = '128'
  )

  if (-not $Model)             { throw "No model. Pass -Model to Probe or when dot-sourcing." }
  if (-not (Test-Path $Model)) { throw "Model not found: $Model" }

  $exe     = Join-Path $Global:LCB_DIR 'llama-batched-bench.exe'
  $slug    = ($Label -replace '[^\w\.\-]', '_')
  $logFile = Join-Path $Global:LCB_LOGS "$slug.txt"
  $smiFile = Join-Path $env:TEMP "vram_$slug.csv"

  $sampler = Start-VramSampler -Path $smiFile
  $sw      = [Diagnostics.Stopwatch]::StartNew()

  # -c is passed here rather than by the caller so it can never be forgotten.
  $argv = @('-m', $Model, '-c', "$Ctx", '-npp', $Npp, '-ntg', $Ntg, '-npl', '1') +
          ($A | Where-Object { $_ -ne '-c' })
  $out  = & $exe @argv 2>&1 | ForEach-Object { [string]$_ }

  $secs = [math]::Round($sw.Elapsed.TotalSeconds, 1)
  $peak = Stop-VramSampler -Job $sampler -Path $smiFile

  # Save evidence BEFORE parsing, so a parse failure cannot destroy the run.
  $out | Out-File -Encoding utf8 $logFile

  $pp = ''; $tg = ''; $status = 'OK'; $notes = @()
  foreach ($l in $out) {
    if ($l -match '^\|\s*\d+\s*\|') {
      $c = ($l -split '\|') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
      if ($c.Count -ge 8) { $pp = $c[5]; $tg = $c[7] }
    }
    if     ($l -match 'out of memory')                   { $status = 'OOM' }
    elseif ($l -match 'resource allocation failed')      { $status = 'OOM' }   # cublasCreate = OOM in disguise
    elseif ($l -match 'failed to (load|create)')          { if ($status -eq 'OK') { $status = 'FAIL' } }
    if     ($l -match 'retrying without pipeline')       { $notes += 'NO-PIPELINE-FALLBACK' }
    if     ($l -match 'overrides to CPU')                { $notes += 'CPU-TENSORS' }
  }
  # A run that produced a result row SUCCEEDED, even if an allocation failed
  # earlier and llama.cpp retried without pipeline parallelism. Labelling those
  # OOM makes consumers discard valid data -- caught by an external review.
  if ($pp -ne '' -and $status -eq 'OOM') { $status = 'OK'; $notes += 'RECOVERED-AFTER-OOM' }
  if ($pp -eq '' -and $status -eq 'OK') { $status = 'FAIL' }

  $p0 = if ($peak.ContainsKey(0)) { $peak[0] } else { 0 }
  $p1 = if ($peak.ContainsKey(1)) { $peak[1] } else { 0 }
  $pt = ($peak.Values | Measure-Object -Sum).Sum

  # Flush this single run before the next one starts.
  Add-Content -Path $Global:LCB_TSV -Value (@(
    (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'),
    $Suite, $Label, $status, (Split-Path $Model -Leaf), $Ctx, $Npp, $Ntg,
    $pp, $tg, $p0, $p1, $pt, $secs, ($A -join ' ')
  ) -join "`t")

  Write-Host ("{0,-30} {1,-4} PP={2,9} TG={3,8}  VRAM {4,5}/{5,5}={6,5} MiB  {7,6}s {8}" -f `
    $Label, $status, $pp, $tg, $p0, $p1, $pt, $secs, ($notes -join ' '))

  if ($notes -contains 'NO-PIPELINE-FALLBACK') {
    Write-Host "   ^ fell back to non-pipelined execution: this is NOT the config you asked for." -ForegroundColor Yellow
    Write-Host "     If it is fast, that IS the fast path here. To make it deliberate rather" -ForegroundColor Yellow
    Write-Host "     than accidental, rebuild with -DLLAMA_SCHED_MAX_COPIES=1 (compile-time;" -ForegroundColor Yellow
    Write-Host "     the environment variable of that name does nothing)." -ForegroundColor Yellow
  }
  if ($notes -contains 'CPU-TENSORS') {
    Write-Host "   ^ tensors on CPU: expect a large prefill penalty on MoE models." -ForegroundColor Yellow
  }

  return [pscustomobject]@{
    Label = $Label; Status = $status; PP = $pp; TG = $tg
    V0 = $p0; V1 = $p1; VT = $pt; Secs = $secs; Notes = ($notes -join ' '); Args = ($A -join ' ')
  }
}

function Show-BenchResults {
  param([int] $Top = 25)
  Import-Csv -Delimiter "`t" $Global:LCB_TSV |
    Where-Object { $_.status -eq 'OK' } |
    Sort-Object { [double]$_.pp_ts } -Descending |
    Select-Object -First $Top suite, label, ctx, npp, pp_ts, tg_ts, vram_total |
    Format-Table -AutoSize
  Write-Host "Full data: $Global:LCB_TSV"
}

Write-Host "harness ready -- results: $Global:LCB_TSV" -ForegroundColor DarkGray
Write-Host "Reminder: differences under ~8% are noise when a GPU also drives a display." -ForegroundColor DarkGray


