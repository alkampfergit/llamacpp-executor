# ===========================================================================
#  needle-test.ps1 -- a long-context QUALITY gate for KV cache settings
#
#  WHY THIS EXISTS
#  Throughput benchmarks cannot detect a quality regression. Quantising the KV
#  cache to q4_0 makes prefill ~24% faster, and a benchmark will happily report
#  that as a win even if the model has started forgetting the middle of your
#  document. So before adopting an aggressive KV setting you need a test that
#  fails when RECALL degrades, not when speed does.
#
#  THE METHOD ("needle in a haystack")
#  Build a very long, deliberately boring document -- 3000 near-identical
#  telemetry records. Hide one unique, unguessable fact in RECORD 0001, right at
#  the START, which is the hardest place to retain: it is furthest from the
#  question and the first thing a degraded cache loses. Then ask for that fact
#  at the very END.
#
#      RECORD 0001: The commissioning passphrase for reactor bay 7
#                   is CRIMSON-PELICAN-4417.
#      RECORD 0002..3000: routine telemetry, no anomalies ...
#      Question: what is the commissioning passphrase for reactor bay 7?
#
#  The passphrase is arbitrary nonsense on purpose. It cannot be inferred, it
#  cannot be guessed, and it does not appear in training data -- so a correct
#  answer proves the model actually retrieved it across ~108,000 tokens of
#  near-duplicate filler. Any other answer is a fail.
#
#  Total prompt: ~107,700 tokens, which exercises most of a 130k window.
#
#  IMPORTANT: use the CHAT endpoint. On a raw /completion an instruction-tuned
#  model emits EOS immediately and returns nothing -- which looks exactly like
#  a quality failure but is a request-format bug. This cost a debugging cycle.
#
#  Usage:
#    ./needle-test.ps1 -Label q8-130k -Ctk q8_0 -Ctv q8_0 -Ub 512
#    ./needle-test.ps1 -Label q4-130k -Ctk q4_0 -Ctv q4_0 -Ub 1024
# ===========================================================================
param(
  [Parameter(Mandatory=$true)][string] $Label,
  [string] $Ctk = 'q8_0',
  [string] $Ctv = 'q8_0',
  [int]    $Ub  = 512,
  [int]    $Ctx = 130048,
  [string] $Split = '12,29',
  [int]    $Port = 9077,
  [Parameter(Mandatory=$true)][string] $Model,
  [int]    $Records = 3000,
  [string] $OutDir,

  # Which build to measure. See Find-LlamaDir below -- without this the walk-up
  # silently selects the BASELINE binaries even when you meant a fresh build.
  [string] $LlamaDir
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
$LC       = Find-LlamaDir -Explicit $LlamaDir
# Portable default is bench-results/ next to the binaries; pass -OutDir to land this
# repo's own campaign in wiki/benchmarks/ instead (see CLAUDE.md).
$benchDir = if ($OutDir) { $OutDir } else { Join-Path $LC 'bench-results' }
$needle   = Join-Path $benchDir 'needle-prompt.txt'
$out      = Join-Path $benchDir 'needle-results.md'
$slog     = Join-Path $benchDir "logs\server_$Label.log"
$SECRET   = 'CRIMSON-PELICAN-4417'

New-Item -ItemType Directory -Force -Path (Join-Path $benchDir 'logs') | Out-Null

# --- build the haystack once and reuse it, so every config sees the SAME prompt
if (-not (Test-Path $needle)) {
  $sb = [System.Text.StringBuilder]::new()
  [void]$sb.AppendLine('=== TECHNICAL LOG ARCHIVE ===')
  [void]$sb.AppendLine("RECORD 0001: The commissioning passphrase for reactor bay 7 is $SECRET.")
  for ($i = 2; $i -le $Records; $i++) {
    $n = $i.ToString('0000')
    [void]$sb.AppendLine("RECORD ${n}: routine telemetry sample; coolant nominal; pressure 4.$($i % 100) bar; operator shift $($i % 7); no anomalies recorded during this interval.")
  }
  [void]$sb.AppendLine('=== END ARCHIVE ===')
  [void]$sb.AppendLine('Question: what is the commissioning passphrase for reactor bay 7? Answer with only the passphrase.')
  $sb.ToString() | Out-File -Encoding utf8 $needle
  Write-Host "built haystack: $needle" -ForegroundColor DarkGray
}

Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 600

$argv = @('-m',$Model,'--host','127.0.0.1','--port',"$Port",'-c',"$Ctx",'-np','1',
          '-ngl','999','-sm','layer','-ts',$Split,'-fa','on','-b','2048','-ub',"$Ub",
          '-ctk',$Ctk,'-ctv',$Ctv,'-fit','off','--temp','0','--no-warmup')

Write-Host "[$Label] server: ctx=$Ctx ub=$Ub kv=$Ctk/$Ctv ts=$Split" -ForegroundColor Cyan
$p = Start-Process -FilePath (Join-Path $LC 'llama-server.exe') -WorkingDirectory $LC -ArgumentList $argv `
       -PassThru -WindowStyle Hidden -RedirectStandardError $slog

$ready = $false
foreach ($i in 1..120) {
  Start-Sleep -Seconds 2
  try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3).status -eq 'ok') { $ready = $true; break } } catch {}
  if ($p.HasExited) { break }
}
if (-not $ready) {
  Write-Host "[$Label] SERVER FAILED TO START -- config does not fit" -ForegroundColor Red
  Get-Content $slog -Tail 12 | Where-Object { $_ -notmatch 'unused tensor' }
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
  return
}

$body = @{
  messages             = @(@{ role = 'user'; content = (Get-Content $needle -Raw) })
  max_tokens           = 32
  temperature          = 0
  chat_template_kwargs = @{ enable_thinking = $false }   # answer directly, no reasoning block
} | ConvertTo-Json -Depth 6 -Compress

Write-Host "[$Label] sending haystack..." -ForegroundColor DarkGray
$sw = [Diagnostics.Stopwatch]::StartNew()
try {
  $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
        -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 900
} catch {
  Write-Host "[$Label] REQUEST FAILED: $_" -ForegroundColor Red
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; return
}
$wall = [math]::Round($sw.Elapsed.TotalSeconds,1)

# A quality gate must not crash, and must not report FAIL for a response it could
# not read -- that conflates "the model missed the needle" with "the harness could
# not parse the payload", which is the same mislabelling trap as calling a
# fallback-recovered run OOM. `$r.choices[0]` throws "Cannot index into a null
# array" when `choices` is absent, and @($null).Count is 1, so the null check is
# load-bearing. Unusable payload -> explicit ERROR, never a silent absence.
$answer   = ''
$unusable = $true
if ($r.PSObject.Properties['choices'] -and $null -ne $r.choices -and @($r.choices).Count -gt 0) {
  $content = $r.choices[0].message.content
  if ($null -ne $content) {
    $answer   = (([string]$content) -replace '\s+',' ').Trim()
    $unusable = $false
  }
}
$verdict = if ($unusable) { 'ERROR' }
           elseif ($answer -match 'CRIMSON[-\s]?PELICAN[-\s]?4417') { 'PASS' }
           else { 'FAIL' }
if ($unusable) {
  Write-Host "[$Label] response carried no usable choices[].message.content -- verdict ERROR, not FAIL" `
    -ForegroundColor Yellow
}
$ppN     = if ($r.timings.prompt_n) { $r.timings.prompt_n } else { $r.usage.prompt_tokens }

Write-Host ("[$Label] needle={0}  depth={1}  PP={2} t/s  TG={3} t/s  wall={4}s" -f `
  $verdict,$ppN,[math]::Round($r.timings.prompt_per_second,1),
  [math]::Round($r.timings.predicted_per_second,1),$wall) `
  -ForegroundColor $(if($verdict -eq 'PASS'){'Green'}else{'Red'})
Write-Host ("[$Label] answer: {0}" -f $answer)

if (-not (Test-Path $out)) {
  "# Long-context needle test results`n" | Out-File -Encoding utf8 $out
  "Retrieve ``$SECRET`` planted in RECORD 0001, asked at ~108k tokens.`n" | Add-Content $out
  "| label | KV | ub | ctx | depth | needle | PP t/s | TG t/s | wall s | answer |" | Add-Content $out
  "|---|---|---|---|---|---|---|---|---|---|" | Add-Content $out
}
("| {0} | {1}/{2} | {3} | {4} | {5} | **{6}** | {7} | {8} | {9} | {10} |" -f `
  $Label,$Ctk,$Ctv,$Ub,$Ctx,$ppN,$verdict,
  [math]::Round($r.timings.prompt_per_second,1),
  [math]::Round($r.timings.predicted_per_second,1),$wall,
  ($answer -replace '\|','/')) | Add-Content $out

Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 600
Write-Host "results: $out" -ForegroundColor DarkGray

