# ===========================================================================
#  mtp-test.ps1 -- measure speculative decoding (MTP / n-gram) over HTTP
#
#  WHY A SEPARATE SCRIPT: neither llama-bench nor llama-batched-bench supports
#  --spec-type, so speculative decoding CANNOT be measured with them. It must
#  be measured against llama-server, and it is workload-dependent -- acceptance
#  rate depends on what is being generated, so a synthetic prompt misleads.
#
#  This uses a GENERATION-HEAVY request (short prompt, long code output) because
#  speculative decoding only accelerates generation, never prefill.
#
#  Usage:
#    ./mtp-test.ps1                        # baseline + draft-mtp n-max 1..6
#    ./mtp-test.ps1 -SpecTypes ngram-simple
# ===========================================================================
param(
  [int]      $Ctx    = 65536,
  [int]      $Ubatch = 512,
  [string]   $Ctk    = 'q4_0',
  [string]   $Ctv    = 'q4_0',
  [string]   $Split  = '12,29',
  [int]      $Port   = 9078,
  [int[]]    $NMax   = @(1,2,3,4,5,6),
  [string[]] $SpecTypes = @('draft-mtp'),
  [int]      $Predict = 400,

  # 0 = greedy. Use 0 whenever you need to resolve a small effect: at the 0.6
  # default the sampled token sequence -- and therefore draft acceptance, and
  # therefore TG -- changes on every run, swamping anything below roughly +-20%.
  [double]   $Temperature = 0.6,
  [Parameter(Mandatory=$true)][string] $Model,
  [string]   $OutDir,

  # Which build to measure. The server is always launched with its
  # WorkingDirectory set here, because ggml resolves backend DLLs relative to
  # the cwd: launching a new build from the baseline root silently loads the
  # baseline's ggml-rpc.dll and ggml-cpu-haswell.dll.
  [string]   $LlamaDir
)

# Locate the llama-server.exe to measure. Same contract as bench-harness.ps1 and
# vram-budget.ps1: -LlamaDir wins, otherwise walk up from this script. NOTE that
# the walk-up lands on the BASELINE root in this repo, so pass -LlamaDir to
# measure a fresh build, e.g. -LlamaDir S:/OsDevelop/llamacpp/llama.cpp/build-mmvq/bin
function Find-LlamaDir {
  param([string] $Explicit)
  if ($Explicit) {
    $r = (Resolve-Path $Explicit -ErrorAction SilentlyContinue).Path
    if ($r -and (Test-Path (Join-Path $r 'llama-server.exe'))) { return $r }
    throw "llama-server.exe not found in -LlamaDir '$Explicit'"
  }
  $dir = $PSScriptRoot
  for ($i = 0; $i -lt 6 -and $dir; $i++) {
    if (Test-Path (Join-Path $dir 'llama-server.exe')) { return $dir }
    $dir = Split-Path $dir -Parent
  }
  throw "Could not locate llama-server.exe. Pass -LlamaDir explicitly."
}
$LC   = Find-LlamaDir -Explicit $LlamaDir
# Portable default is bench-results/ next to the binaries; pass -OutDir to land this
# repo's own campaign in wiki/benchmarks/ instead (see CLAUDE.md).
$bd   = if ($OutDir) { $OutDir } else { Join-Path $LC 'bench-results' }
# Two sweeps that differ only by an env var would otherwise write the same log
# filenames and the first arm's logs would be silently overwritten.
$runTag = if ($env:GGML_CUDA_MMVQ_MAX_BATCH) { "mmvq$($env:GGML_CUDA_MMVQ_MAX_BATCH)" } else { 'mmvqdef' }
$out  = Join-Path $bd 'mtp-results.md'
New-Item -ItemType Directory -Force -Path (Join-Path $bd 'logs') | Out-Null

# A prompt whose OUTPUT is long and code-shaped: the regime where drafting wins,
# because source code is highly predictable (brackets, repeated identifiers).
$PROMPT = @'
Write a complete Python module implementing a thread-safe LRU cache.
Include: full type hints, a decorator interface, TTL expiry, hit/miss statistics,
and docstrings for every public method. Output only code.
'@

function Measure-Spec {
  param([string]$Label, [string[]]$Extra)

  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
  Start-Sleep -Milliseconds 600
  $slog = Join-Path $bd "logs\mtp_${runTag}_$($Label -replace '[^\w\.\-]','_').log"

  $argv = @('-m',$Model,'--host','127.0.0.1','--port',"$Port",'-c',"$Ctx",'-np','1',
            '-ngl','999','-sm','layer','-ts',$Split,'-fa','on','-b','2048','-ub',"$Ubatch",
            '-ctk',$Ctk,'-ctv',$Ctv,'-fit','off','--no-warmup') + $Extra

  $p = Start-Process -FilePath (Join-Path $LC 'llama-server.exe') -WorkingDirectory $LC -ArgumentList $argv `
         -PassThru -WindowStyle Hidden -RedirectStandardError $slog

  $ready = $false
  foreach ($i in 1..90) {
    Start-Sleep -Seconds 2
    try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3).status -eq 'ok') { $ready=$true; break } } catch {}
    if ($p.HasExited) { break }
  }
  if (-not $ready) {
    Write-Host ("{0,-24} SERVER FAILED (config does not fit?)" -f $Label) -ForegroundColor Red
    Get-Content $slog -Tail 6 | Where-Object { $_ -notmatch 'unused tensor' }
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
    return $null
  }

  # Did the MTP head actually load? If its tensors are still "unused ...
  # ignoring", the draft head is NOT active and any speedup is imaginary.
  #
  # Match the nextn tensors, NOT a hardcoded block index. The MTP layer sits at
  # blk.<n_layer>, which is model-specific: blk.40 on a 40-layer model, blk.64
  # on Qwen3.8-27B. A hardcoded index reports "MTP loaded = True" for EVERY run
  # on any other model, baseline included.
  $mtpIgnored = (Select-String -Path $slog -Pattern 'unused tensor blk\.\d+\.nextn' -Quiet)

  $body = @{ messages = @(@{role='user'; content=$PROMPT})
             max_tokens = $Predict; temperature = $Temperature; top_p = 0.95
             chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 6 -Compress

  # one warm request (loads caches), then the measured one
  try { Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
          -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 600 | Out-Null } catch {}
  try {
    $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
          -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 600
  } catch {
    Write-Host ("{0,-24} REQUEST FAILED: {1}" -f $Label,$_) -ForegroundColor Red
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; return $null
  }

  $tg = [math]::Round($r.timings.predicted_per_second,2)
  $n  = $r.timings.predicted_n

  # Guard the extraction. `$r.choices[0]` on a success payload that carries no
  # `choices` throws "Cannot index into a null array" -- verified, not assumed --
  # and it would throw ABOVE the Stop-Process below, leaking a 16-21 GiB model
  # into VRAM so that every remaining point in the sweep then fails to fit.
  # Losing one row's hash is the cheap outcome; degrade to '-' and carry on.
  $txt = ''
  try {
    # `$null -ne $r.choices` is load-bearing: @($null).Count is 1, not 0, so a
    # Count check alone lets a null `choices` through to the indexing.
    if ($r.PSObject.Properties['choices'] -and $null -ne $r.choices -and @($r.choices).Count -gt 0) {
      $txt = [string]$r.choices[0].message.content
    }
  } catch {
    Write-Host ("{0,-24} note: no usable choices[] in response; sha unavailable" -f $Label) `
      -ForegroundColor DarkYellow
  }
  $sha = if ($txt) {
    $h = [Security.Cryptography.SHA256]::Create()
    try { (([BitConverter]::ToString($h.ComputeHash([Text.Encoding]::UTF8.GetBytes($txt)))) -replace '-','').Substring(0,8) }
    finally { $h.Dispose() }
  } else { '-' }

  # Acceptance and mean accepted run come from the server's own print_timing line.
  # Both requests log one, so take the LAST (the measured one, not the warm-up).
  # Without these you cannot tell a kernel-cost change from an acceptance change.
  $acc = $null; $mlen = $null
  $accLine = Select-String -Path $slog -Pattern 'draft acceptance =\s*([\d.]+).*?mean len =\s*([\d.]+)' |
             Select-Object -Last 1
  if ($accLine) {
    $acc  = [double]$accLine.Matches[0].Groups[1].Value
    $mlen = [double]$accLine.Matches[0].Groups[2].Value
  }
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
  Start-Sleep -Milliseconds 500

  # Only draft-mtp needs the nextn tensors loaded. n-gram drafters use no model weights, so
  # "unused tensor blk.<n>.nextn" is correct and expected for them -- do not warn.
  $flag = if ($mtpIgnored -and $Label -like '*draft-mtp*') { '  <-- MTP TENSORS STILL IGNORED!' } else { '' }
  Write-Host ("{0,-24} TG={1,7} t/s   tokens={2,4}{3}" -f $Label,$tg,$n,$flag) `
    -ForegroundColor $(if($flag){'Yellow'}else{'Green'})

  return [pscustomobject]@{ Label=$Label; TG=$tg; Tokens=$n; MtpIgnored=$mtpIgnored
                            Acc=$acc; MeanLen=$mlen; Sha=$sha }
}

# Label every number with the binary that produced it (CLAUDE.md precondition 3) and with
# any env var that changes kernel selection -- otherwise two sweeps that differ only by an
# env var produce two identical-looking tables.
$verLine = (& (Join-Path $LC 'llama-server.exe') --version 2>&1 |
            Select-String 'version:' | Select-Object -First 1).Line
if (-not $verLine) { $verLine = '(--version produced no version line)' }
$envNote = if ($env:GGML_CUDA_MMVQ_MAX_BATCH) { "GGML_CUDA_MMVQ_MAX_BATCH=$($env:GGML_CUDA_MMVQ_MAX_BATCH)" }
           else { 'GGML_CUDA_MMVQ_MAX_BATCH unset' }

Write-Host "=== Speculative decoding: ctx=$Ctx ub=$Ubatch KV=$Ctk/$Ctv predict=$Predict ===" -ForegroundColor Cyan
Write-Host "bin: $LC" -ForegroundColor DarkGray
Write-Host "    $($verLine.Trim())" -ForegroundColor DarkGray
Write-Host "    $envNote" -ForegroundColor DarkGray
$rows = @()
$rows += Measure-Spec 'baseline' @()
foreach ($st in $SpecTypes) {
  foreach ($n in $NMax) {
    $rows += Measure-Spec "$st n$n" @('--spec-type',$st,'--spec-draft-n-max',"$n")
  }
}
$rows = $rows | Where-Object { $_ }

$base = ($rows | Where-Object { $_.Label -eq 'baseline' }).TG
if (-not (Test-Path $out)) { "# Speculative decoding results`n" | Out-File -Encoding utf8 $out }
("`n## ctx=$Ctx ub=$Ubatch KV=$Ctk/$Ctv predict=$Predict ts=$Split temp=$Temperature`n") | Add-Content $out
("- bin: ``$LC``") | Add-Content $out
("- $($verLine.Trim())") | Add-Content $out
("- $envNote`n") | Add-Content $out
"| config | TG t/s | vs baseline | acceptance | mean accepted | out sha | MTP loaded |" | Add-Content $out
"|---|---|---|---|---|---|---|" | Add-Content $out
foreach ($r in $rows) {
  $sp = if ($base -gt 0) { "{0:P0}" -f (($r.TG - $base)/$base) } else { '-' }
  ("| {0} | {1} | {2} | {3} | {4} | {5} | {6} |" -f $r.Label,$r.TG,$sp,
     $(if($null -ne $r.Acc){$r.Acc}else{'-'}), $(if($null -ne $r.MeanLen){$r.MeanLen}else{'-'}),
     $r.Sha, (-not $r.MtpIgnored)) | Add-Content $out
}
Write-Host "`nbaseline TG = $base t/s" -ForegroundColor Cyan
Write-Host "results: $out" -ForegroundColor DarkGray


