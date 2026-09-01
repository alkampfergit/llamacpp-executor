# ===========================================================================
#  vram-budget.ps1 -- predict which llama.cpp configurations fit, in seconds
#
#  Reads only the GGUF header via llama-fit-params, so it never loads weights.
#  Prints a -c x -ub feasibility grid per KV type, plus a suggested -ts and a
#  shortlist of candidates worth actually benchmarking.
#
#  RUN this script; do not read it. Its job is arithmetic, not instruction.
#
#  Examples:
#    ./vram-budget.ps1 -Model S:/models/foo.gguf
#    ./vram-budget.ps1 -Model S:/models/foo.gguf -CtxList 65536,131072 -KvTypes q8_0,q4_0
# ===========================================================================
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)] [string] $Model,

  # Directory holding llama-fit-params.exe. Auto-detected by walking up from
  # this script, then falling back to PATH.
  [string]   $LlamaDir,

  # Spare VRAM to leave free per GPU, in MiB, in device order. A single value
  # is broadcast. Leave more on whichever GPU drives the display: its usage
  # fluctuates while your model is loaded.
  [int[]]    $Margins = @(300, 1024),

  [int[]]    $CtxList  = @(8192, 32768, 65536, 98304, 131072),
  [int[]]    $UbList   = @(256, 512, 1024),

  # Extra MiB per device added to every requirement. Default 0: calibration
  # against measured runs showed llama-fit-params' TOTAL is already slightly
  # conservative (a config it costed at 22426 MiB actually used ~21827), even
  # though an individual compute buffer can be under-predicted. Padding on top
  # of that double-counts and hides working configurations. Raise it only if
  # your own runs OOM where this script predicted a fit.
  [int]      $PadPerDevice = 0,

  # Only symmetric pairs with compiled flash-attention kernels are useful:
  # f16, bf16, q8_0, q4_0. Anything else leaves the GPU. See known-traps.md.
  [string[]] $KvTypes = @('q8_0', 'f16')
)

$ErrorActionPreference = 'Stop'

# --- locate the binary -----------------------------------------------------
function Find-LlamaDir {
  param([string] $Explicit)
  if ($Explicit) {
    if (Test-Path (Join-Path $Explicit 'llama-fit-params.exe')) { return $Explicit }
    throw "llama-fit-params.exe not found in -LlamaDir '$Explicit'"
  }
  $dir = $PSScriptRoot
  for ($i = 0; $i -lt 6 -and $dir; $i++) {
    if (Test-Path (Join-Path $dir 'llama-fit-params.exe')) { return $dir }
    $dir = Split-Path $dir -Parent
  }
  $onPath = Get-Command 'llama-fit-params.exe' -ErrorAction SilentlyContinue
  if ($onPath) { return Split-Path $onPath.Source -Parent }
  throw "Could not locate llama-fit-params.exe. Pass -LlamaDir explicitly."
}

$LlamaDir = Find-LlamaDir -Explicit $LlamaDir
$fit      = Join-Path $LlamaDir 'llama-fit-params.exe'
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }

# --- 1. the real budget ----------------------------------------------------
# Installed VRAM is not available VRAM. Whatever the desktop already holds is
# gone, and it is usually concentrated on the display GPU.
$gpus = @()
foreach ($line in (nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits)) {
  $p = $line -split ',\s*'
  if ($p.Count -ge 4) {
    $gpus += [pscustomobject]@{
      Index = [int]$p[0]; Name = $p[1].Trim()
      Total = [int]$p[2]; Used = [int]$p[3]
      Free  = [int]$p[2] - [int]$p[3]
    }
  }
}
if (-not $gpus) { throw "nvidia-smi returned no GPUs." }

for ($i = 0; $i -lt $gpus.Count; $i++) {
  $m = if ($i -lt $Margins.Count) { $Margins[$i] } else { $Margins[-1] }
  $gpus[$i] | Add-Member -NotePropertyName Margin -NotePropertyValue $m -Force
  $gpus[$i] | Add-Member -NotePropertyName Usable -NotePropertyValue ([Math]::Max(0, $gpus[$i].Free - $m)) -Force
}

Write-Host "`n=== GPU budget (MiB) ===" -ForegroundColor Cyan
# Render via Out-String: a bare Format-Table emits format objects that get
# mangled if the caller pipes this script's output anywhere.
Write-Host (($gpus | Format-Table Index, Name, Total, Used, Free, Margin, Usable -AutoSize | Out-String).TrimEnd())
$budgetTotal = ($gpus | Measure-Object -Property Usable -Sum).Sum   # free minus safety margin
$freeTotal   = ($gpus | Measure-Object -Property Free   -Sum).Sum   # physical ceiling
Write-Host ("Free across GPUs: {0} MiB   |   usable after margins: {1} MiB" -f $freeTotal, $budgetTotal)
if (($gpus | Measure-Object -Property Used -Sum).Sum -gt 1000) {
  Write-Host "NOTE: >1 GiB is already held by other processes. Closing GPU-using desktop" -ForegroundColor Yellow
  Write-Host "      apps is the only way to free VRAM at zero speed cost." -ForegroundColor Yellow
}

# --- 2. model shape --------------------------------------------------------
$meta = & $fit -m $Model -v 2>&1 | ForEach-Object { [string]$_ }
function Meta-Int { param([string]$Key)
  $m = $meta | Select-String -Pattern "$Key\s*=\s*(\d+)" | Select-Object -First 1
  if ($m) { return [int]$m.Matches[0].Groups[1].Value }
  return 0
}
$nLayerAll = Meta-Int 'n_layer_all'
$nLayer    = Meta-Int 'n_layer'
$nCtxTrain = Meta-Int 'n_ctx_train'
$nExpert   = Meta-Int 'n_expert'
$layers    = if ($nLayerAll -gt 0) { $nLayerAll } else { $nLayer }
$isHybrid  = [bool]($meta | Select-String -Pattern 'ssm_(d_conv|n_group)')

Write-Host "`n=== Model ===" -ForegroundColor Cyan
Write-Host ("file        : {0}" -f (Split-Path $Model -Leaf))
Write-Host ("size        : {0} GiB" -f [math]::Round((Get-Item $Model).Length / 1GB, 2))
Write-Host ("layers      : {0} (n_layer={1}, n_layer_all={2})" -f $layers, $nLayer, $nLayerAll)
Write-Host ("n_ctx_train : {0}   <- quality ceiling, do not exceed" -f $nCtxTrain)
if ($nExpert -gt 0) { Write-Host ("MoE         : {0} experts -- CPU offload will wreck prefill" -f $nExpert) }
if ($isHybrid)      { Write-Host  "hybrid attn : yes -- KV cache far smaller than the standard formula predicts" }

# --- 3. suggested tensor split --------------------------------------------
# Proportional to USABLE free memory, not to capacity. Then benchmark +/-2
# layers: the optimum is a memory boundary and cannot be computed exactly.
if ($gpus.Count -gt 1 -and $layers -gt 0 -and $budgetTotal -gt 0) {
  $parts = foreach ($g in $gpus) { [math]::Max(1, [math]::Round($layers * $g.Usable / $budgetTotal)) }
  $tsSuggest = ($parts -join ',')
  Write-Host ("`nSuggested -ts : {0}   (proportional to usable free VRAM)" -f $tsSuggest) -ForegroundColor Green

  # Proportional-to-memory OVER-allocates to the smaller GPU, because weights
  # split by the -ts ratio but compute buffers do not: every device carries a
  # near-full one. On the reference box this arithmetic suggested 15 where the
  # measured optimum was 12. So sweep DOWNWARD from the suggestion.
  if ($gpus.Count -eq 2) {
    $lo = [math]::Max(1, $parts[0] - 4)
    $sweep = @()
    for ($n = $parts[0] + 1; $n -ge $lo; $n--) { $sweep += "$n,$($layers - $n)" }
    Write-Host ("Sweep these -ts values (measured optima sit 2-4 layers BELOW the suggestion):") -ForegroundColor Green
    Write-Host ("  " + ($sweep -join '   ')) -ForegroundColor DarkGray
    # Bias the emitted example toward the empirically-likely winner.
    $tsSuggest = "$([math]::Max(1, $parts[0] - 3)),$($layers - [math]::Max(1, $parts[0] - 3))"
  }
} else { $tsSuggest = $null }

# --- 4. feasibility grid ---------------------------------------------------
function Get-Requirement {
  param([int]$Ctx, [int]$Ub, [string]$Kv)
  $a = @('-m', $Model, '-c', "$Ctx", '-fa', 'on', '-ub', "$Ub", '-fitp', 'on')
  if ($Kv -ne 'f16') { $a += @('-ctk', $Kv, '-ctv', $Kv) }
  $out = & $fit @a 2>$null
  $gpuTot = 0; $hostTot = 0; $per = @{}
  foreach ($l in $out) {
    $p = ($l -split '\s+') | Where-Object { $_ -ne '' }
    if ($p.Count -ge 4) {
      $sum = 0
      for ($i = 1; $i -le 3; $i++) { $v = 0; if ([int]::TryParse($p[$i], [ref]$v)) { $sum += $v } }
      if ($p[0] -like 'CUDA*') { $gpuTot += $sum; $per[$p[0]] = $sum }
      elseif ($p[0] -eq 'Host') { $hostTot = $sum }
    }
  }
  return [pscustomobject]@{ Gpu = $gpuTot; Host = $hostTot; Per = $per }
}

$candidates = @()
foreach ($kv in $KvTypes) {
  Write-Host "`n=== Feasibility: KV = $kv/$kv (fa on) ===" -ForegroundColor Cyan
  $rows = @()
  foreach ($c in $CtxList) {
    foreach ($u in $UbList) {
      $r = Get-Requirement -Ctx $c -Ub $u -Kv $kv
      if ($r.Gpu -le 0) { continue }

      $needed = $r.Gpu + ($PadPerDevice * $gpus.Count)

      # Two slack figures, because they answer different questions:
      #   vsFree   -- will it physically load at all?
      #   vsSafe   -- will it still load once the desktop grows?
      $slackFree = $freeTotal   - $needed
      $slackSafe = $budgetTotal - $needed

      # A large Host figure means tensors could not stay on the GPU at all --
      # usually an unsupported KV pair. Flag it rather than calling it a fit.
      $offGpu = $r.Host -gt 1500

      # Calibrated against measured runs on a 3070+5060 Ti box: every config
      # that actually loaded lands in FITS/TIGHT/MARGINAL, and every config
      # that OOM'd lands in NO FIT. MARGINAL means "fits physically but eats
      # the safety margin" -- which is exactly where the best 130k config sat,
      # so it must stay on the shortlist rather than being filtered out.
      $verdict = if     ($offGpu)             { 'OFF-GPU'  }
                 elseif ($slackSafe -ge 400)  { 'FITS'     }
                 elseif ($slackSafe -ge 0)    { 'TIGHT'    }
                 elseif ($slackFree -ge 0)    { 'MARGINAL' }
                 else                         { 'NO FIT'   }

      $rows += [pscustomobject]@{
        ctx = $c; ub = $u; kv = $kv; needGPU = $needed
        vsFree = $slackFree; vsSafe = $slackSafe; host = $r.Host; verdict = $verdict
      }
      if ($verdict -in 'FITS', 'TIGHT', 'MARGINAL') {
        $candidates += [pscustomobject]@{
          ctx = $c; ub = $u; kv = $kv; vsFree = $slackFree; vsSafe = $slackSafe; verdict = $verdict }
      }
    }
  }
  Write-Host (($rows | Format-Table ctx, ub, needGPU, vsFree, vsSafe, host, verdict -AutoSize | Out-String).TrimEnd())
}

# --- 5. shortlist ----------------------------------------------------------
Write-Host "`n=== Candidates worth benchmarking ===" -ForegroundColor Green
if (-not $candidates) {
  Write-Host "NONE fit. Work down the ladder (cheapest first):" -ForegroundColor Yellow
  Write-Host "  1. close GPU-using desktop apps          (free)"
  Write-Host "  2. q8_0/q8_0 KV, then q4_0/q4_0 KV       (both on the fast FA path)"
  Write-Host "  3. q8_0/q8_0 KV, then q4_0/q4_0          (verify quality)"
  Write-Host "  4. reduce -c                             (also speeds prefill up)"
  Write-Host "  5. lower -ub                             (~-30% prefill)"
  Write-Host "  6. -ncmoe N -- LAST RESORT                (~-44% prefill for 2 layers)"
} else {
  # Prefer the largest ubatch at each context: prefill peaks around ub 512 and
  # generation is indifferent, so ub is the axis that actually buys throughput.
  $short = $candidates |
    Group-Object ctx |
    ForEach-Object { $_.Group | Sort-Object ub -Descending | Select-Object -First 2 } |
    Sort-Object @{e = 'ctx'; Descending = $true}, @{e = 'ub'; Descending = $true}
  Write-Host (($short | Format-Table ctx, ub, kv, vsFree, vsSafe, verdict -AutoSize | Out-String).TrimEnd())

  Write-Host "Verdicts (vsFree = spare vs physical free VRAM; vsSafe = vs free minus margins):" -ForegroundColor DarkGray
  Write-Host "  FITS     >=400 MiB spare even after margins -- safe" -ForegroundColor DarkGray
  Write-Host "  TIGHT    fits within margins, but under 400 MiB spare" -ForegroundColor DarkGray
  Write-Host "  MARGINAL fits physically, but EATS THE SAFETY MARGIN. Expect it to load" -ForegroundColor DarkGray
  Write-Host "           and run, and to fail the day you open something on the display GPU." -ForegroundColor DarkGray
  Write-Host "           The fastest long-context config often lives here -- benchmark it." -ForegroundColor DarkGray
  Write-Host "  NO FIT   exceeds physical free VRAM. Work the ladder before retrying." -ForegroundColor DarkGray

  # Largest context first: that is normally the binding requirement, and the
  # point of measuring is to find the fastest config that still meets it.
  $best  = $short | Select-Object -First 1
  $tsArg = if ($tsSuggest) { "'-ts','$tsSuggest'," } else { '' }
  Write-Host "`nNext: measure with the harness (fresh process per run, results appended)." -ForegroundColor Green
  Write-Host '  # do not raise -ub merely to provoke a scheduler allocation failure.' -ForegroundColor DarkGray
  Write-Host '  # a successful retry saves memory; benchmark tensor split and ubatch separately.' -ForegroundColor DarkGray
  Write-Host '  . ./bench-harness.ps1 -Model <model.gguf>' -ForegroundColor DarkGray
  Write-Host ("  Probe 'c{0} ub{1}' @('-ub','{1}','-ctk','{2}','-ctv','{2}',{3}'-fa','on','-ngl','999','-fit','off') -Ctx {0}" `
    -f $best.ctx, $best.ub, $best.kv, $tsArg) -ForegroundColor DarkGray
}


