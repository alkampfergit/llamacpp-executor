# ===========================================================================
#  build-llamacpp.ps1 -- build llama.cpp with CUDA on Windows, reliably
#
#  RUN this script; do not read it and hand-type the steps. The generator
#  choice, the single-process environment, and the CUDA architecture list are
#  all load-bearing -- see SKILL.md and references/windows-toolchain-traps.md.
#
#  It performs, in order:
#    1. locate vcvars64.bat (preferring a real VS2022 over any preview)
#    2. locate ninja, and say how to install it if missing
#    3. derive CUDA architectures from nvidia-smi (never guessed)
#    4. clear the build dir if the generator would change
#    5. write ONE .bat that sources vcvars64 and runs configure + build,
#       because env vars set by a batch file do not survive a new process
#    6. smoke-test the result, including the CUDA-runtime-DLL PATH trap
#
#  Examples:
#    ./build-llamacpp.ps1
#    ./build-llamacpp.ps1 -CMakeExtra '-DGGML_CUDA_FA_ALL_QUANTS=ON','-DLLAMA_SCHED_MAX_COPIES=1'
#    ./build-llamacpp.ps1 -Targets llama-server -BuildDir build-fast
# ===========================================================================
[CmdletBinding()]
param(
  # Defaults to the llama.cpp submodule beside this skill, if present.
  [string]   $Repo,

  # Empty = build everything. Naming targets is much faster when iterating.
  [string[]] $Targets = @(),

  [string]   $BuildDir = 'build',

  # Extra -D options, e.g. -DGGML_CUDA_FA_ALL_QUANTS=ON
  [string[]] $CMakeExtra = @(),

  # Override only if nvidia-smi is unavailable; format "86;120".
  [string]   $CudaArch,

  # Wipe the build dir first (full rebuild).
  [switch]   $Clean,

  [switch]   $CpuOnly
)

$ErrorActionPreference = 'Stop'

function Fail { param([string]$m) Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }
function Step { param([string]$m) Write-Host "`n=== $m ===" -ForegroundColor Cyan }

# --- 0. repo ---------------------------------------------------------------
if (-not $Repo) {
  $d = $PSScriptRoot
  for ($i = 0; $i -lt 6 -and $d; $i++) {
    if (Test-Path (Join-Path $d 'llama.cpp\CMakeLists.txt')) { $Repo = Join-Path $d 'llama.cpp'; break }
    if (Test-Path (Join-Path $d 'CMakeLists.txt'))           { $Repo = $d; break }
    $d = Split-Path $d -Parent
  }
}
if (-not $Repo -or -not (Test-Path (Join-Path $Repo 'CMakeLists.txt'))) {
  Fail "No llama.cpp source tree found. Pass -Repo <path to llama.cpp checkout>."
}
$Repo = (Resolve-Path $Repo).Path
Step "Source tree"
Write-Host $Repo
Push-Location $Repo
try { git log -1 --format='  branch: %D%n  commit: %h %s' 2>&1 | Write-Host } catch {}
Pop-Location

# --- 1. vcvars64.bat -------------------------------------------------------
# Prefer a genuine VS2022 (v170). A VS 2026 preview ("18") is exactly what
# breaks nvcc here, so it is used only as a last resort.
Step "Visual Studio environment"
$vcvars = $null
$roots = @()
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
  $roots += & $vswhere -all -products '*' -property installationPath 2>$null
}
$roots += @(
  "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise",
  "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional",
  "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community"
)
# 2022 first, previews last.
$ordered = @($roots | Where-Object { $_ -match '\\2022\\' }) + @($roots | Where-Object { $_ -notmatch '\\2022\\' })
foreach ($r in ($ordered | Where-Object { $_ } | Select-Object -Unique)) {
  $cand = Join-Path $r 'VC\Auxiliary\Build\vcvars64.bat'
  if (Test-Path $cand) { $vcvars = $cand; break }
}
if (-not $vcvars) { Fail "vcvars64.bat not found. Install VS2022 with the 'Desktop development with C++' workload." }
Write-Host $vcvars
if ($vcvars -notmatch '\\2022\\') {
  Write-Host "WARNING: this is not a VS2022 install. nvcc's cudafe++ is known to crash" -ForegroundColor Yellow
  Write-Host "         (ACCESS_VIOLATION 0xC0000005) against preview MSVC toolsets." -ForegroundColor Yellow
}

# --- 2. ninja --------------------------------------------------------------
Step "Ninja generator"
$ninja = (Get-Command ninja -ErrorAction SilentlyContinue).Source
if (-not $ninja) {
  Fail @"
ninja not found, and it is required.

Ninja is NOT bundled with Visual Studio or CMake. It is used deliberately: it
calls cl.exe and nvcc.exe directly, so CUDA's MSBuild integration -- which the
CUDA installer often registers only for a preview VS -- is never needed.

Install it in user space (no admin, nothing written to Program Files):
    pip install ninja
"@
}
Write-Host $ninja

# --- 3. CUDA architectures -------------------------------------------------
Step "CUDA architectures"
if ($CpuOnly) {
  Write-Host "CPU-only build requested; skipping."
} else {
  if (-not $CudaArch) {
    try {
      # Sort NUMERICALLY: a string sort yields "120;86", which is valid but
      # reads wrong and makes review harder. EVERY installed card must appear,
      # or the omitted one silently falls back to JIT-from-PTX at runtime.
      $caps = nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null |
              ForEach-Object { $_.Trim() -replace '\.', '' } |
              Where-Object { $_ } | Sort-Object { [int]$_ } -Unique
      if ($caps) {
        $CudaArch = ($caps -join ';')
        Write-Host "Detected $($caps.Count) distinct architecture(s) across $((nvidia-smi -L | Measure-Object).Count) GPU(s)."
      }
    } catch {}
  }
  if (-not $CudaArch) { Fail "Could not derive CUDA architectures. Pass -CudaArch '86;120'." }
  Write-Host "-DCMAKE_CUDA_ARCHITECTURES=`"$CudaArch`""
  nvidia-smi --query-gpu=index,name,compute_cap --format=csv | Write-Host
  # A wrong arch is SILENT: the binary works but JIT-compiles from PTX.
  if ($CudaArch -match '\b89\b' -and $CudaArch -notmatch '\b120\b') {
    Write-Host "NOTE: 89 is Ada. Blackwell cards (RTX 50xx) are 120 -- verify against the table above." -ForegroundColor Yellow
  }
}

# --- 4. build dir / generator consistency ---------------------------------
$bd = Join-Path $Repo $BuildDir
if ($Clean -and (Test-Path $bd)) {
  Step "Cleaning $BuildDir"
  Remove-Item $bd -Recurse -Force
}
$cache = Join-Path $bd 'CMakeCache.txt'
if (Test-Path $cache) {
  $gen = (Select-String -Path $cache -Pattern '^CMAKE_GENERATOR:INTERNAL=(.*)$').Matches.Groups[1].Value
  if ($gen -and $gen -ne 'Ninja') {
    # CMake refuses to reconfigure across generators; the only fix is a wipe.
    Step "Existing build dir used generator '$gen' -- wiping (CMake cannot switch generators)"
    Remove-Item $bd -Recurse -Force
  }
}

# --- 5. configure + build, in ONE process --------------------------------
Step "Configure and build"
$cmakeArgs = @("-B `"$BuildDir`"", '-G Ninja', '-DCMAKE_BUILD_TYPE=Release')
if (-not $CpuOnly) { $cmakeArgs += @('-DGGML_CUDA=ON', "-DCMAKE_CUDA_ARCHITECTURES=`"$CudaArch`"") }
$cmakeArgs += '-DGGML_CCACHE=OFF'
$cmakeArgs += $CMakeExtra
$buildArgs = @("--build `"$BuildDir`"", '-j')
foreach ($t in $Targets) { $buildArgs += "--target $t" }

Write-Host "cmake $($cmakeArgs -join ' ')"
Write-Host "cmake $($buildArgs -join ' ')"
if ($CMakeExtra -match 'FA_ALL_QUANTS') {
  Write-Host "GGML_CUDA_FA_ALL_QUANTS=ON: expect a MUCH longer compile and a larger binary." -ForegroundColor Yellow
}

$bat = Join-Path $env:TEMP "llamacpp-build-$PID.bat"
@"
@echo off
call "$vcvars"
if errorlevel 1 exit /b 1
cd /d "$Repo"
cmake $($cmakeArgs -join ' ')
if errorlevel 1 exit /b 2
cmake $($buildArgs -join ' ')
if errorlevel 1 exit /b 3
"@ | Out-File -FilePath $bat -Encoding ascii

$sw = [Diagnostics.Stopwatch]::StartNew()
$log = Join-Path $env:TEMP "llamacpp-build-$PID.log"
& cmd /c "`"$bat`"" 2>&1 | Tee-Object -FilePath $log | ForEach-Object {
  # Keep the console readable: progress lines and real diagnostics only.
  if ($_ -match '^\[\d+/\d+\]|^-- |error|Error|FAILED|fatal') { Write-Host $_ }
}
$code = $LASTEXITCODE
$mins = [math]::Round($sw.Elapsed.TotalMinutes, 1)
Remove-Item $bat -Force -ErrorAction SilentlyContinue

if ($code -ne 0) {
  Write-Host "`nBUILD FAILED (exit $code) after $mins min. Full log: $log" -ForegroundColor Red
  Write-Host "Last 25 lines:" -ForegroundColor Red
  Get-Content $log -Tail 25
  Write-Host "`nMatch the error against references/windows-toolchain-traps.md before changing flags." -ForegroundColor Yellow
  exit 1
}
Write-Host "`nBuild OK in $mins min. Log: $log" -ForegroundColor Green

# --- 6. smoke test -------------------------------------------------------
# Ninja is single-config, so binaries are flat in build/bin -- no Release/.
Step "Smoke test"
$bin = Join-Path $bd 'bin'
if (-not (Test-Path $bin)) { Fail "No $bin directory. (Note: Ninja does NOT create bin\Release\.)" }
Get-ChildItem $bin -Filter *.exe | Select-Object Name,
  @{n='KB';e={[math]::Round($_.Length/1KB)}} | Format-Table -AutoSize | Out-String | Write-Host

$probe = Join-Path $bin 'llama-bench.exe'
if (-not (Test-Path $probe)) { $probe = (Get-ChildItem $bin -Filter *.exe | Select-Object -First 1).FullName }
if ($probe -and (Test-Path $probe)) {
  # The silent-exit trap: CUDA 13.x puts its runtime DLLs in bin\x64, but only
  # bin is usually on PATH, so the binary dies with no message at all.
  $cudaX64 = Get-ChildItem "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
             Sort-Object Name -Descending | ForEach-Object { Join-Path $_.FullName 'bin\x64' } |
             Where-Object { Test-Path $_ } | Select-Object -First 1
  if ($cudaX64 -and $env:PATH -notlike "*$cudaX64*") {
    Write-Host "Adding CUDA runtime dir to PATH for this test: $cudaX64" -ForegroundColor DarkGray
    $env:PATH = "$cudaX64;$env:PATH"
  }
  $out = & $probe --version 2>&1 | ForEach-Object { [string]$_ }
  if (-not $out) {
    Write-Host "SMOKE TEST FAILED: no output at all." -ForegroundColor Red
    Write-Host "That is the missing-CUDA-runtime-DLL signature, not a build error." -ForegroundColor Yellow
    Write-Host "Put the CUDA bin\x64 directory on PATH, or copy cudart64_*/cublas*64_* next to the exe." -ForegroundColor Yellow
  } else {
    $out | Select-Object -First 8 | Write-Host
    $devs = ($out | Select-String 'Device \d+:' | Measure-Object).Count
    if ($devs -gt 0) { Write-Host "OK: $devs CUDA device(s) visible to the new build." -ForegroundColor Green }
    elseif (-not $CpuOnly) { Write-Host "WARNING: built with CUDA but no devices listed." -ForegroundColor Yellow }
  }
}

Write-Host "`nBinaries: $bin" -ForegroundColor Green
Write-Host "Do not overwrite a known-good binary set until you have re-verified throughput." -ForegroundColor DarkGray
