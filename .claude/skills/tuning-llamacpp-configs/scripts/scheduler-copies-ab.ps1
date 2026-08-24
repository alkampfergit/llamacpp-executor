[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)] [string] $Model,
  [Parameter(Mandatory = $true)] [string] $ControlDir,
  [Parameter(Mandatory = $true)] [string] $SingleCopyDir,
  [Parameter(Mandatory = $true)] [string] $OutDir,
  [int] $Ctx = 32768,
  [int] $Ubatch = 512,
  [string] $TensorSplit = '14,27',
  [int] $Repetitions = 3,
  [switch] $ExpectControlFallback
)

$harnessPath = Join-Path $PSScriptRoot 'bench-harness.ps1'
$runResults = [Collections.Generic.List[object]]::new()

function Invoke-SchedulerArm {
  param(
    [string] $ArmName,
    [string] $BinaryDir,
    [int] $Rep
  )

  . $harnessPath -Model $Model -LlamaDir $BinaryDir -OutDir $OutDir
  $probeArgs = @(
    '-ngl', '999', '-sm', 'layer', '-ts', $TensorSplit,
    '-fa', 'on', '-b', '2048', '-ub', "$Ubatch", '-fit', 'off',
    '-ctk', 'q8_0', '-ctv', 'q8_0'
  )
  $result = Probe "$ArmName r$Rep" $probeArgs -Suite 'scheduler-copies-ab' -Ctx $Ctx
  $result | Add-Member -NotePropertyName Arm -NotePropertyValue $ArmName
  $runResults.Add($result)
  if ($result.Status -ne 'OK') {
    throw "$ArmName r$Rep did not produce a result row. Stop the campaign and inspect its log."
  }
}

if (-not (Test-Path $Model))        { throw "Model not found: $Model" }
if (-not (Test-Path $ControlDir))   { throw "Control directory not found: $ControlDir" }
if (-not (Test-Path $SingleCopyDir)){ throw "Single-copy directory not found: $SingleCopyDir" }
if ($Repetitions -lt 2)             { throw 'Use at least two repetitions per arm.' }

for ($rep = 1; $rep -le $Repetitions; $rep++) {
  Invoke-SchedulerArm -ArmName 'control-max4' -BinaryDir $ControlDir -Rep $rep
  Invoke-SchedulerArm -ArmName 'single-copy-max1' -BinaryDir $SingleCopyDir -Rep $rep
}

$controlRuns = @($runResults | Where-Object Arm -eq 'control-max4')
$singleRuns = @($runResults | Where-Object Arm -eq 'single-copy-max1')

if ($ExpectControlFallback -and $controlRuns.Notes -notmatch 'NO-PIPELINE-FALLBACK') {
  Write-Warning 'The control did not fall back, so this campaign did not compare runtime fallback with the single-copy build.'
} elseif (-not $ExpectControlFallback -and $controlRuns.Notes -match 'NO-PIPELINE-FALLBACK') {
  Write-Warning 'The control fell back. This campaign did not isolate scheduler copy count.'
}
if ($singleRuns.ExecutionMode -ne 'single-copy-build') {
  Write-Warning 'The treatment CMake cache did not identify GGML_SCHED_MAX_COPIES=1.'
}

$culture = [Globalization.CultureInfo]::InvariantCulture
$controlMean = ($controlRuns | ForEach-Object { [double]::Parse($_.PP, $culture) } |
  Measure-Object -Average).Average
$singleMean = ($singleRuns | ForEach-Object { [double]::Parse($_.PP, $culture) } |
  Measure-Object -Average).Average
$delta = 100 * ($singleMean / $controlMean - 1)

Write-Host ''
Write-Host ('control max-copies=4 mean PP : {0:N1} t/s' -f $controlMean)
Write-Host ('single-copy build mean PP   : {0:N1} t/s' -f $singleMean)
Write-Host ('single-copy delta           : {0:+0.0;-0.0;0.0}%' -f $delta)
Write-Host "evidence                    : $OutDir"
