# ===========================================================================
#  check-vram-residency.ps1 -- prove whether a llama-server configuration is
#  ENTIRELY resident in VRAM, or is silently spilling into system RAM.
#
#  Why this exists: since NVIDIA driver 536.40 a CUDA allocation past VRAM does
#  not fail on Windows. The driver places the overflow in host RAM and runs it
#  over PCIe. nvidia-smi CANNOT see this -- it reports dedicated VRAM only, and
#  a spilling process looks merely "slow".
#
#  The WDDM performance counters CAN see it, per PROCESS and per adapter:
#      \GPU Process Memory(pid_<pid>_luid_...)\Dedicated Usage   <- real VRAM
#      \GPU Process Memory(pid_<pid>_luid_...)\Non Local Usage   <- SYSTEM RAM
#      \GPU Process Memory(pid_<pid>_luid_...)\Shared Usage      <- system RAM
#  "Non local" is WDDM's term for memory that is not local to the GPU.
#
#  DO NOT read a raw non-local figure as a spill. llama.cpp legitimately keeps
#  several hundred MiB of host buffers (compare against llama-fit-params' `Host`
#  row, which this script prints). A real spill on a display GPU can be SMALLER
#  than that budget and still cost 26% of prefill -- it shows up as a DELTA
#  against a matched control, and in the adapter-wide figure. Always compare two
#  runs that differ in one variable. See wiki/19-vram-residency.md.
#
#  Usage:
#    ./check-vram-residency.ps1 -Model <path.gguf> -Ctx 130000 -Ts 12,29
#    ./check-vram-residency.ps1 -Model <path.gguf> -Ctx 64000  -Ts 12,29
#  Compare the two. See wiki/19-vram-residency.md.
# ===========================================================================
[CmdletBinding()]
param(
  [Parameter(Mandatory)][string] $Model,
  [int]      $Ctx             = 130000,
  [string]   $Ts              = '12,29',
  [int]      $Ubatch          = 512,
  [int]      $Batch           = 2048,
  [string]   $Ctk             = 'q8_0',
  [string]   $Ctv             = 'q8_0',
  [int]      $Np              = 1,
  [int]      $Port            = 9099,
  [int]      $PromptTokens    = 8192,
  [int]      $Reps            = 2,
  [int]      $ReadyTimeoutSec = 900,
  [string]   $Label,
  [string]   $Bin,
  [string]   $OutDir,
  [string[]] $ExtraArgs       = @(),
  [switch]   $KeepMmap
)

$ErrorActionPreference = 'Stop'
if (-not $Label) { $Label = "c$Ctx-ts$($Ts -replace ',','_')-ub$Ubatch-$Ctk" }

# --- locate llama-server.exe -------------------------------------------------
if ($Bin) {
  $LC = (Resolve-Path $Bin).Path
} else {
  $LC = $PSScriptRoot
  while ($LC -and -not (Test-Path (Join-Path $LC 'llama-server.exe'))) { $LC = Split-Path $LC -Parent }
}
if (-not $LC -or -not (Test-Path (Join-Path $LC 'llama-server.exe'))) {
  throw "llama-server.exe not found (searched parents of $PSScriptRoot; use -Bin to override)"
}
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }
if (-not $OutDir) { $OutDir = Join-Path $LC 'bench-results' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$log  = Join-Path $OutDir "residency-$Label.log"
$json = Join-Path $OutDir "residency-$Label.json"

Write-Host "llama-server : $LC" -ForegroundColor DarkGray
Write-Host "results      : $OutDir" -ForegroundColor DarkGray

function Get-ProcCounters([int]$ServerPid) {
  $res = @{}
  foreach ($name in @('Dedicated Usage', 'Non Local Usage', 'Shared Usage')) {
    $samples = $null
    try { $samples = (Get-Counter "\GPU Process Memory(*)\$name" -ErrorAction Stop).CounterSamples } catch { }
    foreach ($s in ($samples | Where-Object { $_.InstanceName -like "pid_${ServerPid}_luid_*" })) {
      $luid = ($s.InstanceName -replace "^pid_${ServerPid}_", '')
      if (-not $res.ContainsKey($luid)) { $res[$luid] = @{} }
      $res[$luid][$name] = [double]$s.CookedValue
    }
  }
  return $res
}

function Get-SmiUsed {
  $rows = & nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits 2>$null
  $o = @{}
  foreach ($r in $rows) {
    $p = $r -split '\s*,\s*'
    if ($p.Count -ge 3) { $o[[int]$p[0]] = @{ total = [int]$p[1]; used = [int]$p[2] } }
  }
  return $o
}

$smiIdle = Get-SmiUsed
Write-Host ""
Write-Host "Idle VRAM before launch:" -ForegroundColor Cyan
foreach ($i in ($smiIdle.Keys | Sort-Object)) {
  Write-Host ("  GPU{0}  {1,6} / {2,6} MiB used" -f $i, $smiIdle[$i].used, $smiIdle[$i].total)
}

# --- launch ------------------------------------------------------------------
$argv = @(
  '-m', $Model, '--host', '127.0.0.1', '--port', "$Port",
  '-c', "$Ctx", '-np', "$Np", '-ctk', $Ctk, '-ctv', $Ctv,
  '-ngl', '999', '-sm', 'layer', '-ts', $Ts,
  '-fa', 'on', '-b', "$Batch", '-ub', "$Ubatch", '-fit', 'off'
)
if (-not $KeepMmap) { $argv += '--no-mmap' }
$argv += $ExtraArgs

Write-Host ""
Write-Host "Launching: llama-server.exe $($argv -join ' ')" -ForegroundColor DarkGray
$proc = Start-Process -FilePath (Join-Path $LC 'llama-server.exe') -WorkingDirectory $LC `
  -ArgumentList $argv -PassThru -WindowStyle Hidden `
  -RedirectStandardError $log -RedirectStandardOutput "$log.out"

$peak    = @{}
$smiPeak = @{}
$adapterShared = @{}
function Update-Peak {
  $c = Get-ProcCounters $proc.Id
  foreach ($luid in $c.Keys) {
    if (-not $peak.ContainsKey($luid)) {
      $peak[$luid] = @{ 'Dedicated Usage' = 0.0; 'Non Local Usage' = 0.0; 'Shared Usage' = 0.0 }
    }
    foreach ($k in $c[$luid].Keys) {
      if ($c[$luid][$k] -gt $peak[$luid][$k]) { $peak[$luid][$k] = $c[$luid][$k] }
    }
  }
  $s = Get-SmiUsed
  foreach ($i in $s.Keys) {
    if (-not $smiPeak.ContainsKey($i)) { $smiPeak[$i] = @{ total = $s[$i].total; used = 0 } }
    if ($s[$i].used -gt $smiPeak[$i].used) { $smiPeak[$i].used = $s[$i].used }
  }
  # Adapter-wide, ALL processes. A display GPU starved of VRAM evicts *other*
  # processes' surfaces (the desktop compositor's), which never shows up in
  # llama-server's own per-process counters but does cost it GPU time.
  $ad = $null
  try { $ad = (Get-Counter '\GPU Adapter Memory(*)\Shared Usage' -ErrorAction Stop).CounterSamples } catch { }
  foreach ($a in $ad) {
    if (-not $adapterShared.ContainsKey($a.InstanceName)) { $adapterShared[$a.InstanceName] = 0.0 }
    if ($a.CookedValue -gt $adapterShared[$a.InstanceName]) { $adapterShared[$a.InstanceName] = [double]$a.CookedValue }
  }
}

# --- wait for readiness ------------------------------------------------------
Write-Host "Waiting for model load" -NoNewline
$deadline = (Get-Date).AddSeconds($ReadyTimeoutSec)
$ready = $false
while ((Get-Date) -lt $deadline) {
  if ($proc.HasExited) { break }
  try {
    $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3 -ErrorAction Stop
    if ($h.status -eq 'ok') { $ready = $true; break }
  } catch { }
  Update-Peak
  Start-Sleep -Milliseconds 700
  Write-Host "." -NoNewline
}
Write-Host ""
if (-not $ready) {
  Write-Host "SERVER DID NOT BECOME READY. Tail of $log :" -ForegroundColor Red
  if (Test-Path $log) { Get-Content $log -Tail 40 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkRed } }
  if (-not $proc.HasExited) { $proc | Stop-Process -Force }
  throw "llama-server failed to start for label '$Label'"
}
Update-Peak
Write-Host "Loaded." -ForegroundColor Green

# --- deterministic prompt ----------------------------------------------------
$para = 'The quick brown fox jumps over the lazy dog while the maintenance crew ' +
        'catalogues every spare component in the northern storage annex and files ' +
        'the resulting inventory under the appropriate departmental reference code. '
$target = [int]($PromptTokens * 3.6)
$sb = [System.Text.StringBuilder]::new()
while ($sb.Length -lt $target) { [void]$sb.Append($para) }
$basePrompt = $sb.ToString()

$results = @()
for ($r = 1; $r -le $Reps; $r++) {
  # A unique prefix defeats LCP prompt-cache reuse, so every rep is a real prefill.
  $prompt = "Request $r of $Reps, nonce $([guid]::NewGuid()). $basePrompt`nReply with the single word OK."
  $body = @{
    messages     = @(@{ role = 'user'; content = $prompt })
    n_predict    = 8
    temperature  = 0
    cache_prompt = $false
  } | ConvertTo-Json -Depth 5 -Compress

  $uri = "http://127.0.0.1:$Port/v1/chat/completions"
  $job = Start-Job -ScriptBlock {
    param($u, $b)
    Invoke-RestMethod -Uri $u -Method Post -Body $b -ContentType 'application/json' -TimeoutSec 1800
  } -ArgumentList $uri, $body

  Write-Host "rep $r/$Reps prefilling" -NoNewline
  while ($job.State -eq 'Running') { Update-Peak; Write-Host "." -NoNewline; Start-Sleep -Milliseconds 600 }
  Write-Host ""
  $resp = Receive-Job $job -ErrorAction SilentlyContinue
  Remove-Job $job -Force
  Update-Peak

  $t = $resp.timings
  if ($t) {
    $results += [pscustomobject]@{
      rep         = $r
      prompt_n    = $t.prompt_n
      prompt_ms   = [math]::Round($t.prompt_ms, 1)
      prefill_tps = [math]::Round($t.prompt_per_second, 1)
      gen_tps     = [math]::Round($t.predicted_per_second, 2)
    }
    Write-Host ("  prompt_n={0}  prefill={1} t/s  gen={2} t/s" -f `
      $t.prompt_n, [math]::Round($t.prompt_per_second, 1), [math]::Round($t.predicted_per_second, 2)) -ForegroundColor Green
  } else {
    Write-Host "  no timings returned" -ForegroundColor Yellow
  }
}

# --- shut down ---------------------------------------------------------------
if (-not $proc.HasExited) { $proc | Stop-Process -Force }
Start-Sleep -Milliseconds 800

# llama-server logs everything to stderr, so the stdout redirect file is always
# empty. Start-Process requires a separate path for it; drop it rather than
# littering the evidence directory with zero-byte files.
if ((Test-Path "$log.out") -and ((Get-Item "$log.out").Length -eq 0)) {
  Remove-Item "$log.out" -Force -ErrorAction SilentlyContinue
}

# --- map adapter LUIDs to nvidia-smi indices by peak dedicated usage ---------
$luidOrder = @($peak.Keys | Sort-Object { - $peak[$_]['Dedicated Usage'] })
$smiOrder  = @($smiPeak.Keys | Sort-Object { - $smiPeak[$_].used })
$luidName = @{}
for ($i = 0; $i -lt $luidOrder.Count; $i++) {
  if ($i -lt $smiOrder.Count) { $luidName[$luidOrder[$i]] = "GPU$($smiOrder[$i])" }
  else { $luidName[$luidOrder[$i]] = $luidOrder[$i] }
}

# --- expected host budget from llama-fit-params ------------------------------
# CRITICAL CALIBRATION: llama.cpp ALWAYS keeps several hundred MiB of host-visible
# buffers (its own `Host` budget: CPU-side weights + pinned staging). WDDM counts
# those as "Non Local"/"Shared" for the process, so a HEALTHY run reads 480-940 MiB
# non-local on this box. Measured 2026-09-01: the FASTEST config of the day had the
# HIGHEST non-local figure (938 MiB). Non-zero non-local is therefore NOT spill.
# Compare it against what llama-fit-params predicts for Host instead.
$hostPredMiB = 0.0
try {
  $fit = & (Join-Path $LC 'llama-fit-params.exe') -m $Model -c $Ctx -ctk $Ctk -ctv $Ctv `
           -fa on -ngl 999 -sm layer -ts $Ts -b $Batch -ub $Ubatch -np $Np -fitp on 2>&1
  foreach ($line in $fit) {
    if ("$line" -match '^\s*Host\s+(\d+)\s+(\d+)\s+(\d+)') {
      $hostPredMiB = [double]$Matches[1] + [double]$Matches[2] + [double]$Matches[3]
    }
  }
} catch { }

# --- verdict -----------------------------------------------------------------
$MiB = 1MB
$nonLocalMax = 0.0
Write-Host ""
Write-Host "=== $Label ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Per-process GPU memory (peak, llama-server):"
Write-Host ("  {0,-8} {1,14} {2,16} {3,14}" -f 'adapter', 'dedicated', 'NON-LOCAL', 'shared')
foreach ($luid in $luidOrder) {
  $d = [math]::Round($peak[$luid]['Dedicated Usage'] / $MiB, 1)
  $n = [math]::Round($peak[$luid]['Non Local Usage'] / $MiB, 1)
  $s = [math]::Round($peak[$luid]['Shared Usage'] / $MiB, 1)
  # max, NOT sum: the same host buffer is visible on both adapters, so summing
  # double-counts it.
  if ($n -gt $nonLocalMax) { $nonLocalMax = $n }
  Write-Host ("  {0,-8} {1,10} MiB {2,12} MiB {3,10} MiB" -f $luidName[$luid], $d, $n, $s)
}
Write-Host ("  expected host budget (llama-fit-params): {0} MiB" -f [math]::Round($hostPredMiB, 0)) -ForegroundColor DarkGray
Write-Host ""
Write-Host "Adapter-wide shared memory, ALL processes (peak):"
foreach ($k in ($adapterShared.Keys | Sort-Object { - $adapterShared[$_] })) {
  $v = [math]::Round($adapterShared[$k] / $MiB, 1)
  if ($v -lt 1) { continue }
  $nm = if ($luidName.ContainsKey($k)) { $luidName[$k] } else { $k }
  Write-Host ("  {0,-40} {1,8} MiB" -f $nm, $v)
}
Write-Host ""
Write-Host "nvidia-smi dedicated VRAM (peak during run):"
foreach ($i in ($smiPeak.Keys | Sort-Object)) {
  $free = $smiPeak[$i].total - $smiPeak[$i].used
  $col = if ($free -lt 400) { 'Yellow' } else { 'Gray' }
  Write-Host ("  GPU{0}  {1,6} / {2,6} MiB used   headroom {3,6} MiB" -f $i, $smiPeak[$i].used, $smiPeak[$i].total, $free) -ForegroundColor $col
}

$warn = @()
if (Test-Path $log) {
  $warn = @(Select-String -Path $log -Pattern 'out of memory|failed to allocate|retrying without|cudaMalloc failed|unable to allocate' -ErrorAction SilentlyContinue)
}
Write-Host ""
if ($warn.Count -gt 0) {
  Write-Host "Allocation warnings in log:" -ForegroundColor Yellow
  $warn | Select-Object -First 12 | ForEach-Object { Write-Host "  $($_.Line.Trim())" -ForegroundColor Yellow }
} else {
  Write-Host "No allocation warnings in log." -ForegroundColor DarkGray
}

$headroomMin = 999999
foreach ($i in $smiPeak.Keys) {
  $h = $smiPeak[$i].total - $smiPeak[$i].used
  if ($h -lt $headroomMin) { $headroomMin = $h }
}
$excess = $nonLocalMax - $hostPredMiB

# A genuine driver spill has BOTH signatures at once: the card is completely full
# (WDDM had nowhere else to put the allocation) AND non-local far exceeds what
# llama.cpp itself asked the host for. Either one alone is normal.
$verdict =
  if ($headroomMin -lt 96 -and $excess -gt 512) { 'SPILLING' }
  elseif ($excess -gt 512)                      { 'SUSPECT' }
  elseif ($warn.Count -gt 0)                    { 'RESIDENT-FALLBACK' }
  else                                          { 'RESIDENT' }

Write-Host ""
switch ($verdict) {
  'SPILLING' {
    Write-Host ("VERDICT: SPILLING -- {0} MiB non-local vs {1} MiB expected, and only {2} MiB VRAM headroom." -f `
      $nonLocalMax, [math]::Round($hostPredMiB, 0), $headroomMin) -ForegroundColor Red
    Write-Host "         Reduce -c, switch to q4_0/q4_0 KV, lower -ub, or rebalance -ts." -ForegroundColor Red
  }
  'SUSPECT' {
    Write-Host ("VERDICT: SUSPECT -- {0} MiB non-local vs {1} MiB expected, but {2} MiB VRAM headroom remains." -f `
      $nonLocalMax, [math]::Round($hostPredMiB, 0), $headroomMin) -ForegroundColor Yellow
    Write-Host "         Check for an asymmetric/unsupported -ctk/-ctv pair, which moves attention off the GPU." -ForegroundColor Yellow
  }
  'RESIDENT-FALLBACK' {
    Write-Host ("VERDICT: RESIDENT IN VRAM ({0} MiB non-local vs {1} MiB expected host budget)." -f `
      $nonLocalMax, [math]::Round($hostPredMiB, 0)) -ForegroundColor Green
    Write-Host "         Nothing spilled. But the pipelined compute-buffer reservation did not fit and" -ForegroundColor DarkYellow
    Write-Host "         llama.cpp recovered on the single-copy path -- see the warnings above." -ForegroundColor DarkYellow
  }
  default {
    Write-Host ("VERDICT: FULLY RESIDENT IN VRAM ({0} MiB non-local vs {1} MiB expected host budget)." -f `
      $nonLocalMax, [math]::Round($hostPredMiB, 0)) -ForegroundColor Green
  }
}
Write-Host ("         min VRAM headroom across GPUs: {0} MiB" -f $headroomMin) -ForegroundColor DarkGray

if ($results.Count -gt 1) {
  # rep 1 is always cold (allocator warm-up, clock ramp); report it separately.
  $warm = $results | Select-Object -Skip 1
  $mean = ($warm | Measure-Object -Property prefill_tps -Average).Average
  Write-Host ("         prefill: rep1 {0} t/s (cold), reps 2-{1} mean {2} t/s (min {3}, max {4})" -f `
    $results[0].prefill_tps, $results.Count, [math]::Round($mean, 0),
    ($warm | Measure-Object -Property prefill_tps -Minimum).Minimum,
    ($warm | Measure-Object -Property prefill_tps -Maximum).Maximum) -ForegroundColor Cyan
}

$out = [pscustomobject]@{
  label     = $Label
  ctx       = $Ctx
  ts        = $Ts
  ub        = $Ubatch
  ctk       = $Ctk
  ctv       = $Ctv
  non_local_max_mib   = [math]::Round($nonLocalMax, 1)
  host_predicted_mib  = [math]::Round($hostPredMiB, 0)
  non_local_excess_mib = [math]::Round($excess, 1)
  headroom_min_mib    = $headroomMin
  verdict   = $verdict
  adapters  = @($luidOrder | ForEach-Object {
      [pscustomobject]@{
        adapter       = $luidName[$_]
        dedicated_mib = [math]::Round($peak[$_]['Dedicated Usage'] / $MiB, 1)
        non_local_mib = [math]::Round($peak[$_]['Non Local Usage'] / $MiB, 1)
        shared_mib    = [math]::Round($peak[$_]['Shared Usage'] / $MiB, 1)
      } })
  smi_peak  = @($smiPeak.Keys | Sort-Object | ForEach-Object {
      [pscustomobject]@{
        gpu          = $_
        used_mib     = $smiPeak[$_].used
        total_mib    = $smiPeak[$_].total
        headroom_mib = $smiPeak[$_].total - $smiPeak[$_].used
      } })
  adapter_shared_mib = @($adapterShared.Keys | ForEach-Object { [pscustomobject]@{ adapter = $_; shared_mib = [math]::Round($adapterShared[$_]/$MiB,1) } })
  reps      = $results
  alloc_warnings = @($warn | ForEach-Object { $_.Line.Trim() })
  log       = $log
}
$out | ConvertTo-Json -Depth 6 | Set-Content -Path $json -Encoding UTF8
Write-Host ""
Write-Host "JSON: $json" -ForegroundColor DarkGray
