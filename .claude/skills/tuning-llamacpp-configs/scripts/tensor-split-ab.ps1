[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)] [string] $Model,
  [Parameter(Mandatory = $true)] [string] $LlamaDir,
  [Parameter(Mandatory = $true)] [string] $OutDir,
  [Parameter(Mandatory = $true)] [string] $SplitA,
  [Parameter(Mandatory = $true)] [string] $SplitB,
  [int] $Ctx = 8448,
  [int] $Ubatch = 512,
  [int] $Repetitions = 3
)

$harnessPath = Join-Path $PSScriptRoot 'bench-harness.ps1'
$runResults = [Collections.Generic.List[object]]::new()

function Invoke-SplitArm {
  param(
    [string] $ArmName,
    [string] $TensorSplit,
    [int] $Rep
  )

  . $harnessPath -Model $Model -LlamaDir $LlamaDir -OutDir $OutDir
  $probeArgs = @(
    '-ngl', '999', '-sm', 'layer', '-ts', $TensorSplit,
    '-fa', 'on', '-b', '2048', '-ub', "$Ubatch", '-fit', 'off',
    '-ctk', 'q8_0', '-ctv', 'q8_0'
  )
  $result = Probe "$ArmName r$Rep" $probeArgs -Suite 'tensor-split-ab' -Ctx $Ctx
  $result | Add-Member -NotePropertyName Arm -NotePropertyValue $ArmName
  $runResults.Add($result)
  if ($result.Status -ne 'OK') {
    throw "$ArmName r$Rep did not produce a result row. Stop the campaign and inspect its log."
  }
}

if (-not (Test-Path $Model))     { throw "Model not found: $Model" }
if (-not (Test-Path $LlamaDir))  { throw "Binary directory not found: $LlamaDir" }
if ($Repetitions -lt 2)          { throw 'Use at least two repetitions per arm.' }

for ($rep = 1; $rep -le $Repetitions; $rep++) {
  Invoke-SplitArm -ArmName "split-$($SplitA -replace ',', '-')" -TensorSplit $SplitA -Rep $rep
  Invoke-SplitArm -ArmName "split-$($SplitB -replace ',', '-')" -TensorSplit $SplitB -Rep $rep
}

$armAName = "split-$($SplitA -replace ',', '-')"
$armBName = "split-$($SplitB -replace ',', '-')"
$armARuns = @($runResults | Where-Object Arm -eq $armAName)
$armBRuns = @($runResults | Where-Object Arm -eq $armBName)
$culture = [Globalization.CultureInfo]::InvariantCulture
$armAMean = ($armARuns | ForEach-Object { [double]::Parse($_.PP, $culture) } |
  Measure-Object -Average).Average
$armBMean = ($armBRuns | ForEach-Object { [double]::Parse($_.PP, $culture) } |
  Measure-Object -Average).Average
$delta = 100 * ($armBMean / $armAMean - 1)

Write-Host ''
Write-Host ("split $SplitA mean PP : {0:N1} t/s" -f $armAMean)
Write-Host ("split $SplitB mean PP : {0:N1} t/s" -f $armBMean)
Write-Host ("split $SplitB delta   : {0:+0.0;-0.0;0.0}%" -f $delta)
Write-Host "evidence              : $OutDir"

