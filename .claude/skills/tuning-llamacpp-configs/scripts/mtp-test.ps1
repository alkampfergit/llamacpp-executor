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
  [int[]]    $NMax   = @(1,2,3,4,6),
  [string[]] $SpecTypes = @('draft-mtp'),
  [int]      $Predict = 400,
  [Parameter(Mandatory=$true)][string] $Model
)

$LC   = $($d=$PSScriptRoot; for($i=0;$i -lt 6 -and $d;$i++){ if(Test-Path (Join-Path $d 'llama-server.exe')){break}; $d=Split-Path $d -Parent }; if(-not $d){throw 'llama-server.exe not found'}; $d)
$bd   = Join-Path $LC 'bench-results'
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
  $slog = Join-Path $bd "logs\mtp_$($Label -replace '[^\w\.\-]','_').log"

  $argv = @('-m',$Model,'--host','127.0.0.1','--port',"$Port",'-c',"$Ctx",'-np','1',
            '-ngl','999','-sm','layer','-ts',$Split,'-fa','on','-b','2048','-ub',"$Ubatch",
            '-ctk',$Ctk,'-ctv',$Ctv,'-fit','off','--no-warmup') + $Extra

  $p = Start-Process -FilePath (Join-Path $LC 'llama-server.exe') -ArgumentList $argv `
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

  # Did the MTP head actually load? If blk.40 is still "unused ... ignoring",
  # the draft head is NOT active and any speedup is imaginary.
  $mtpIgnored = (Select-String -Path $slog -Pattern 'unused tensor blk\.40' -Quiet)

  $body = @{ messages = @(@{role='user'; content=$PROMPT})
             max_tokens = $Predict; temperature = 0.6; top_p = 0.95
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
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
  Start-Sleep -Milliseconds 500

  # Only draft-mtp needs blk.40 loaded. n-gram drafters use no model weights, so
  # "unused tensor blk.40" is correct and expected for them -- do not warn.
  $flag = if ($mtpIgnored -and $Label -like '*draft-mtp*') { '  <-- MTP TENSORS STILL IGNORED!' } else { '' }
  Write-Host ("{0,-24} TG={1,7} t/s   tokens={2,4}{3}" -f $Label,$tg,$n,$flag) `
    -ForegroundColor $(if($flag){'Yellow'}else{'Green'})

  return [pscustomobject]@{ Label=$Label; TG=$tg; Tokens=$n; MtpIgnored=$mtpIgnored }
}

Write-Host "=== Speculative decoding: ctx=$Ctx ub=$Ubatch KV=$Ctk/$Ctv predict=$Predict ===" -ForegroundColor Cyan
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
("`n## ctx=$Ctx ub=$Ubatch KV=$Ctk/$Ctv predict=$Predict`n") | Add-Content $out
"| config | TG t/s | vs baseline | MTP loaded |" | Add-Content $out
"|---|---|---|---|" | Add-Content $out
foreach ($r in $rows) {
  $sp = if ($base -gt 0) { "{0:P0}" -f (($r.TG - $base)/$base) } else { '-' }
  ("| {0} | {1} | {2} | {3} |" -f $r.Label,$r.TG,$sp,(-not $r.MtpIgnored)) | Add-Content $out
}
Write-Host "`nbaseline TG = $base t/s" -ForegroundColor Cyan
Write-Host "results: $out" -ForegroundColor DarkGray

