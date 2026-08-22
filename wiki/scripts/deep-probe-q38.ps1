# ===========================================================================
#  deep-probe-q38.ps1 -- REAL deep-context numbers over HTTP for Qwen3.8-27B
#
#  llama-batched-bench only ever measures an 8k prompt, and neither bench tool
#  can drive --spec-type at all. This drives llama-server directly and reports:
#    1. shallow TG        (short prompt, 400-token answer)
#    2. deep PP + deep TG (the 107.7k-token needle haystack, long answer)
#    3. prompt-cache reuse on an identical second request (f_sim_best)
#    4. peak VRAM per GPU, sampled throughout
#
#  Usage:
#    ./deep-probe-q38.ps1 -Label q8-130k
#    ./deep-probe-q38.ps1 -Label q8-130k-mtp4 -Spec draft-mtp -NMax 4
# ===========================================================================
param(
  [Parameter(Mandatory=$true)][string] $Label,
  [string] $Model = 'S:\HuggingFace\lmstudio\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf',
  [string] $Ctk = 'q8_0', [string] $Ctv = 'q8_0',
  [int]    $Ub  = 512,    [int]    $Ctx = 130048,
  [string] $Split = '22,43',
  [string] $Spec = '',    [int] $NMax = 0,
  [string] $DraftKv = '',
  [int]    $Port = 9079
)

$LC   = 'S:\OneDrive\Tools\llamacpp'
$bd   = Join-Path $LC 'wiki\benchmarks'
$out  = Join-Path $bd 'deep-results.md'
$slog = Join-Path $bd "logs\deep_$Label.log"
$hay  = Get-Content (Join-Path $bd 'needle-prompt.txt') -Raw

# swap the needle's terse final instruction for one that forces a LONG answer,
# so deep-context TG is measured over enough tokens to mean something.
$hay = $hay -replace 'Question: what is the commissioning passphrase for reactor bay 7\? Answer with only the passphrase\.', `
  'Question: first state the commissioning passphrase for reactor bay 7 exactly. Then write a detailed paragraph of at least 150 words describing what this archive contains and how the records are structured.'

Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 800

$argv = @('-m',$Model,'--host','127.0.0.1','--port',"$Port",'-c',"$Ctx",'-np','1',
          '-ngl','999','-sm','layer','-ts',$Split,'-fa','on','-b','2048','-ub',"$Ub",
          '-ctk',$Ctk,'-ctv',$Ctv,'-fit','off','--temp','0','--no-warmup')
if ($Spec) { $argv += @('--spec-type',$Spec,'--spec-draft-n-max',"$NMax") }
if ($DraftKv) { $argv += @('-ctkd',$DraftKv,'-ctvd',$DraftKv) }

$smi = Join-Path $env:TEMP "deepvram_$Label.csv"
if (Test-Path $smi) { Remove-Item $smi -Force }
$sampler = Start-Job -ScriptBlock { param($p) while($true){ try{ Add-Content -Path $p -Value ((nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits) -join ';') -EA SilentlyContinue }catch{}; Start-Sleep -Milliseconds 400 } } -ArgumentList $smi

$p = Start-Process -FilePath (Join-Path $LC 'llama-server.exe') -ArgumentList $argv -PassThru -WindowStyle Hidden -RedirectStandardError $slog
$ready = $false
foreach ($i in 1..90) { Start-Sleep -Seconds 2
  try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3).status -eq 'ok'){ $ready=$true; break } } catch {}
  if ($p.HasExited) { break } }
if (-not $ready) {
  Write-Host "[$Label] SERVER FAILED -- does not fit" -ForegroundColor Red
  Get-Content $slog -Tail 15 | Where-Object { $_ -notmatch 'unused tensor' }
  Stop-Job $sampler -EA SilentlyContinue; Remove-Job $sampler -Force -EA SilentlyContinue
  Get-Process llama-server -EA SilentlyContinue | Stop-Process -Force; return
}

function Ask($content, $maxTok) {
  $body = @{ messages=@(@{role='user'; content=$content}); max_tokens=$maxTok; temperature=0
             chat_template_kwargs=@{ enable_thinking=$false } } | ConvertTo-Json -Depth 6 -Compress
  return Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
           -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 900
}

$shallow = Ask 'Write a complete Python module implementing a thread-safe LRU cache with type hints, a decorator interface, TTL expiry and hit/miss statistics. Output only code.' 400
Write-Host ("[$Label] shallow: PP={0} t/s over {1} tok | TG={2} t/s over {3} tok" -f `
  [math]::Round($shallow.timings.prompt_per_second,1), $shallow.timings.prompt_n,
  [math]::Round($shallow.timings.predicted_per_second,2), $shallow.timings.predicted_n)

$d1 = Ask $hay 220
Write-Host ("[$Label] deep#1 : PP={0} t/s over {1} tok | TG={2} t/s over {3} tok" -f `
  [math]::Round($d1.timings.prompt_per_second,1), $d1.timings.prompt_n,
  [math]::Round($d1.timings.predicted_per_second,2), $d1.timings.predicted_n)
$needle = if ($d1.choices[0].message.content -match 'CRIMSON[-\s]?PELICAN[-\s]?4417') {'PASS'} else {'FAIL'}
Write-Host "[$Label] needle in long answer: $needle"

$d2 = Ask $hay 220
Write-Host ("[$Label] deep#2 : PP={0} t/s over {1} tok (cache reuse test)" -f `
  [math]::Round($d2.timings.prompt_per_second,1), $d2.timings.prompt_n)
$sim = (Select-String -Path $slog -Pattern 'f_sim_best' | Select-Object -Last 1).Line
if ($sim) { Write-Host "[$Label] $($sim -replace '\x1b\[[0-9;]*m','')" }

$acc = (Select-String -Path $slog -Pattern 'draft acceptance' | Select-Object -Last 1).Line
if ($acc) { Write-Host "[$Label] $($acc -replace '\x1b\[[0-9;]*m','')" }

Get-Process llama-server -EA SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 800
Stop-Job $sampler -EA SilentlyContinue; Remove-Job $sampler -Force -EA SilentlyContinue
$pk=@{}; foreach($l in (Get-Content $smi -EA SilentlyContinue)){ foreach($e in ($l -split ';')){ $q=$e -split ',\s*'; if($q.Count -ge 2){ $i=0;$v=0; if([int]::TryParse($q[0].Trim(),[ref]$i) -and [int]::TryParse($q[1].Trim(),[ref]$v)){ if(-not $pk.ContainsKey($i) -or $v -gt $pk[$i]){$pk[$i]=$v} } } } }
$v0=$pk[0]; $v1=$pk[1]; $vt=($pk.Values|Measure-Object -Sum).Sum
Write-Host ("[$Label] peak VRAM {0}/{1} = {2} MiB" -f $v0,$v1,$vt) -ForegroundColor Cyan

$specDesc = if ($Spec) { "$Spec n$NMax" } else { 'none' }
if ($DraftKv) { $specDesc += " dkv=$DraftKv" }
if (-not (Test-Path $out)) {
  "# Real deep-context results (llama-server over HTTP)`n" | Out-File -Encoding utf8 $out
  "| label | KV | ub | ctx | spec | shallow TG | deep PP | deep TG | cached PP | needle | peak V0 | peak V1 | peak total |" | Add-Content $out
  "|---|---|---|---|---|---|---|---|---|---|---|---|---|" | Add-Content $out
}
("| {0} | {1}/{2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | **{10}** | {11} | {12} | {13} |" -f `
  $Label,$Ctk,$Ctv,$Ub,$Ctx,$specDesc,
  [math]::Round($shallow.timings.predicted_per_second,2),
  [math]::Round($d1.timings.prompt_per_second,1),
  [math]::Round($d1.timings.predicted_per_second,2),
  [math]::Round($d2.timings.prompt_per_second,1),
  $needle,$v0,$v1,$vt) | Add-Content $out
Write-Host "results: $out" -ForegroundColor DarkGray
