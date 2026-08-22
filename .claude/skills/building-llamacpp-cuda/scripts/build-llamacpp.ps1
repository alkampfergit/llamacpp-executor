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
#    ./build-llamacpp.ps1 -CMakeExtra '-DGGML_CUDA_FA_ALL_QUANTS=ON','-DGGML_SCHED_MAX_COPIES=1'
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

  # Which CUDA toolkit to build against. Defaults to the NEWEST installed.
  #
  # This exists because the machine's persisted user environment can point at
  # an older toolkit than the newest installed one: CUDA_PATH and the leading
  # PATH entry are written into the user registry by whichever installer ran,
  # so a newer toolkit can be present and still lose. CMake resolves nvcc from
  # $ENV{CUDA_PATH}/bin and PATH, so it would silently pick the old one.
  # Setting this explicitly makes the choice visible in the build log instead.
  [string]   $CudaToolkit,

  # Wipe the build dir first (full rebuild).
  [switch]   $Clean,

  [switch]   $CpuOnly,

  # Short label for the provenance branch, e.g. 'fa-all-quants'. Defaults to a
  # slug derived from -CMakeExtra, or 'default' when no extra flags are given.
  [string]   $Label,

  # Skip the branch / commit / push provenance step. Use ONLY when offline or
  # when deliberately throwing the build away.
  [switch]   $NoProvenance
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

# --- 2b. CUDA toolkit selection -------------------------------------------
# Pick the toolkit explicitly and inject it into the build environment, rather
# than trusting whatever CUDA_PATH/PATH happen to say. Version-sort numerically:
# a string sort puts v13.1 after v13.3.
$cudaRoot = Join-Path ${env:ProgramFiles} 'NVIDIA GPU Computing Toolkit\CUDA'
if (-not $CpuOnly) {
  Step "CUDA toolkit"
  if (-not $CudaToolkit) {
    $cands = @()
    if (Test-Path $cudaRoot) {
      foreach ($d in (Get-ChildItem $cudaRoot -Directory -ErrorAction SilentlyContinue)) {
        $nv = Join-Path $d.FullName 'bin\nvcc.exe'
        if (Test-Path $nv) {
          $m = [regex]::Match($d.Name, '^v(\d+)\.(\d+)$')
          if ($m.Success) {
            $cands += [pscustomobject]@{
              Path = $d.FullName; Name = $d.Name
              Key  = [int]$m.Groups[1].Value * 1000 + [int]$m.Groups[2].Value
            }
          }
        }
      }
    }
    if ($cands) { $CudaToolkit = ($cands | Sort-Object Key -Descending | Select-Object -First 1).Path }
  }
  if (-not $CudaToolkit -or -not (Test-Path (Join-Path $CudaToolkit 'bin\nvcc.exe'))) {
    Fail "No CUDA toolkit with bin\nvcc.exe found. Pass -CudaToolkit '<path to CUDA\vXX.Y>'."
  }
  $tkVer = (& (Join-Path $CudaToolkit 'bin\nvcc.exe') --version 2>&1 |
            Select-String 'release' | ForEach-Object { $_.Line.Trim() })
  Write-Host "using   : $CudaToolkit"
  Write-Host "nvcc    : $tkVer"

  # Show what would have been picked, so a silent mismatch is visible.
  $onPath = (Get-Command nvcc -ErrorAction SilentlyContinue).Source
  if ($onPath) {
    $onPathVer = (& nvcc --version 2>&1 | Select-String 'release' | ForEach-Object { $_.Line.Trim() })
    if ($onPath -notlike "$CudaToolkit*") {
      Write-Host "NOTE: PATH's nvcc is a DIFFERENT toolkit and is being overridden:" -ForegroundColor Yellow
      Write-Host "      $onPath" -ForegroundColor Yellow
      Write-Host "      $onPathVer" -ForegroundColor Yellow
    }
  }
  if ($env:CUDA_PATH -and $env:CUDA_PATH -ne $CudaToolkit) {
    Write-Host "NOTE: CUDA_PATH currently says '$env:CUDA_PATH' -- overridden for this build." -ForegroundColor Yellow
  }
}

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

# GUARD: never build into, or over, a directory holding baseline binaries.
# The prebuilt executables beside the source tree are a measurement baseline;
# overwriting them silently invalidates every previously recorded benchmark,
# because old and new numbers stop being comparable and the data cannot say so.
$bdFull      = [IO.Path]::GetFullPath($bd).TrimEnd('\')
$repoFull    = [IO.Path]::GetFullPath($Repo).TrimEnd('\')
$parentFull  = [IO.Path]::GetFullPath((Split-Path $Repo -Parent)).TrimEnd('\')
$baselineHit = (Test-Path (Join-Path $bdFull 'llama-server.exe')) -and
               -not (Test-Path (Join-Path $bdFull 'CMakeCache.txt'))
if ($bdFull -eq $repoFull -or $bdFull -eq $parentFull -or $baselineHit) {
  Fail @"
Refusing to build into '$bdFull'.

That directory is the source root, its parent, or already holds llama-*.exe with
no CMakeCache.txt -- i.e. a BASELINE binary set, not a build tree.

Those binaries are immutable: every recorded measurement was produced by them, so
replacing them makes past and future numbers silently incomparable.

Build into llama.cpp's standard output directory and run from there:
    -BuildDir build        ->  $repoFull\build\bin\
Then compare against the baseline by running BOTH and labelling which binary
produced which numbers.
"@
}
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

# --- 4b. provenance: branch + commit + push BEFORE compiling -------------
# Every build gets a named, pushed ref on the fork recording exactly what
# produced it. Without this, a binary in build-*/bin is untraceable a week
# later: you cannot tell which source, flags, compiler or CUDA made it.
#
# Always on a NEW BRANCH, never on master -- master must stay a clean mirror of
# upstream so `git rebase upstream/master` keeps working (see CLAUDE.md).
$provBranch = $null
if (-not $NoProvenance) {
  Step "Build provenance (branch + commit + push)"
  if (-not $Label) {
    $slug = ($CMakeExtra -join ' ') -replace '-D','' -replace '=ON','' -replace '=OFF','off' `
            -replace '[^A-Za-z0-9]+','-' -replace '(^-|-$)',''
    if ($slug.Length -gt 40) { $slug = $slug.Substring(0,40).TrimEnd('-') }
    $Label = if ($slug) { $slug.ToLower() } else { 'default' }
  }
  $stamp      = Get-Date -Format 'yyyyMMdd-HHmmss'
  $provBranch = "build/$stamp-$Label"

  Push-Location $Repo
  try {
    $baseCommit = (git rev-parse --short HEAD).Trim()
    $dirty      = (git status --porcelain | Measure-Object).Count
    if ($dirty -gt 0) {
      Write-Host "NOTE: submodule has $dirty uncommitted change(s); they will be included." -ForegroundColor Yellow
    }
    git checkout -b $provBranch 2>&1 | Write-Host

    # A manifest is what makes the commit meaningful -- it is the only record
    # of the flags, and it lives next to the source that used them.
    $manifest = @"
# Build provenance

| Field | Value |
| --- | --- |
| Branch | ``$provBranch`` |
| Base commit | ``$baseCommit`` |
| Built (local) | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss K') |
| Build dir | ``$BuildDir`` |
| Generator | Ninja |
| CUDA archs | ``$(if($CpuOnly){'n/a (CPU-only)'}else{$CudaArch})`` |
| Extra CMake | ``$(if($CMakeExtra){$CMakeExtra -join ' '}else{'(none - control build)'})`` |
| Targets | ``$(if($Targets){$Targets -join ' '}else{'(all)'})`` |
| Host compiler | $(try { (& "$((Get-Command cl -ErrorAction SilentlyContinue).Source)" 2>&1 | Select-Object -First 1) } catch { 'MSVC (via vcvars64)' }) |
| CUDA toolkit | $((nvcc --version 2>&1 | Select-String 'release').Line.Trim()) |
| GPUs | $((nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader) -join '; ') |

Generated by ``build-llamacpp.ps1`` before compiling. Binaries land in
``$BuildDir/bin/`` and are NOT copied over the baseline binaries in the parent
directory -- those are an immutable measurement reference.
"@
    $manifest | Out-File -Encoding utf8 (Join-Path $Repo 'PROVENANCE.md')
    git add PROVENANCE.md 2>&1 | Out-Null
    git -c commit.gpgsign=false commit -m "build: $Label ($stamp)" -m "Base $baseCommit. Flags: $(if($CMakeExtra){$CMakeExtra -join ' '}else{'defaults'}). Archs: $CudaArch." 2>&1 | Select-Object -First 2 | Write-Host

    Write-Host "pushing $provBranch to origin (your fork)..." -ForegroundColor DarkGray
    $push = git push -u origin $provBranch 2>&1
    if ($LASTEXITCODE -ne 0) {
      Write-Host "PUSH FAILED -- the branch and commit exist locally, so the build can proceed." -ForegroundColor Yellow
      $push | Select-Object -Last 4 | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
    } else {
      Write-Host "pushed: $provBranch" -ForegroundColor Green
    }
  } finally { Pop-Location }

  Write-Host "NOTE: the submodule now points at a new branch, so the PARENT repo sees a" -ForegroundColor DarkGray
  Write-Host "      modified gitlink. Commit that in the parent if you want to record it." -ForegroundColor DarkGray
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
# Pin CUDA_PATH and put the chosen toolkit FIRST on PATH inside the same
# process that runs cmake. CMakeDetermineCUDACompiler searches
# $ENV{CUDA_PATH}/bin and then PATH, so without this it would resolve
# whatever the persisted user environment says -- which may be an older
# toolkit than the one selected above.
$cudaEnv = if ($CpuOnly -or -not $CudaToolkit) { '' } else {
@"
set "CUDA_PATH=$CudaToolkit"
set "CUDA_HOME=$CudaToolkit"
set "PATH=$CudaToolkit\bin;$CudaToolkit\bin\x64;%PATH%"
"@
}
@"
@echo off
call "$vcvars"
if errorlevel 1 exit /b 1
$cudaEnv
cd /d "$Repo"
where nvcc
nvcc --version | findstr /C:"release"
cmake $($cmakeArgs -join ' ')
if errorlevel 1 exit /b 2
cmake $($buildArgs -join ' ')
if errorlevel 1 exit /b 3
"@ | Out-File -FilePath $bat -Encoding ascii

$sw = [Diagnostics.Stopwatch]::StartNew()
$log = Join-Path $env:TEMP "llamacpp-build-$PID.log"
& cmd /c "`"$bat`"" 2>&1 | Tee-Object -FilePath $log | ForEach-Object {
  # Keep the console readable: progress lines and real diagnostics only.
  # 'CMake Warning' and 'not used by the project' are in this filter for a
  # specific reason: a MISSPELLED -D option is only a WARNING. CMake prints
  # "Manually-specified variables were not used by the project: X", compiles
  # happily with the default, and exits 0. Without these patterns the build
  # looks green while silently ignoring the flag you cared about.
  if ($_ -match '^\[\d+/\d+\]|^-- |error|Error|FAILED|fatal|CMake Warning|not used by the project') { Write-Host $_ }
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

# A misspelled -D option does NOT fail the build. Surface it loudly, because
# otherwise you wait out a long compile and get the default behaviour anyway.
$unused = Select-String -Path $log -Pattern 'Manually-specified variables were not used' -Context 0,6
if ($unused) {
  Write-Host "`n!! CMAKE IGNORED ONE OR MORE -D OPTIONS !!" -ForegroundColor Red
  Write-Host "The build succeeded, but these variables were not recognised, so their" -ForegroundColor Yellow
  Write-Host "defaults were compiled in. Check the spelling and the correct PREFIX:" -ForegroundColor Yellow
  Write-Host "ggml options are GGML_*, not LLAMA_* (e.g. GGML_SCHED_MAX_COPIES)." -ForegroundColor Yellow
  $unused | ForEach-Object { $_.Line; $_.Context.PostContext } | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
}

# --- 6. smoke test -------------------------------------------------------
# Ninja is single-config, so binaries are flat in build/bin -- no Release/.
Step "Smoke test"
$bin = Join-Path $bd 'bin'
if (-not (Test-Path $bin)) { Fail "No $bin directory. (Note: Ninja does NOT create bin\Release\.)" }
Get-ChildItem $bin -Filter *.exe | Select-Object Name,
  @{n='KB';e={[math]::Round($_.Length/1KB)}} | Format-Table -AutoSize | Out-String | Write-Host

$probe = Join-Path $bin 'llama-server.exe'   # llama-bench has NO --version flag
if (-not (Test-Path $probe)) { $probe = (Get-ChildItem $bin -Filter *.exe | Select-Object -First 1).FullName }
if ($probe -and (Test-Path $probe)) {
  # The silent-exit trap: CUDA 13.x puts its runtime DLLs in bin\x64, but only
  # bin is usually on PATH, so the binary dies with no message at all.
  # Use the SAME toolkit we built with. Picking it by name-sort could pair a
  # 13.1-built binary with 13.3 runtime DLLs -- it would work (minor-version
  # compatibility) but it is exactly the cross-version mismatch we are trying
  # to eliminate, and it would make the smoke test unrepresentative.
  $cudaX64 = if ($CudaToolkit) { Join-Path $CudaToolkit 'bin\x64' } else {
    Get-ChildItem $cudaRoot -Directory -ErrorAction SilentlyContinue |
      Sort-Object Name -Descending | ForEach-Object { Join-Path $_.FullName 'bin\x64' } |
      Where-Object { Test-Path $_ } | Select-Object -First 1
  }
  if ($cudaX64 -and -not (Test-Path $cudaX64)) { $cudaX64 = $null }
  if ($cudaX64 -and $env:PATH -notlike "*$cudaX64*") {
    Write-Host "Adding CUDA runtime dir to PATH for this test: $cudaX64" -ForegroundColor DarkGray
    $env:PATH = "$cudaX64;$env:PATH"
  }
  # Run FROM the binary's own directory. ggml discovers backend DLLs relative to
  # the working directory, so probing from elsewhere makes the new binary load
  # backends belonging to a DIFFERENT build. Observed for real: a fresh build
  # loaded ggml-rpc.dll and ggml-cpu-haswell.dll out of the baseline folder,
  # because the filenames differ from this build's ggml-cpu.dll. A benchmark run
  # that way would silently mix two builds and be worthless.
  # TWO probes, because they answer different questions:
  #   --version       prints build number + commit, but EXITS BEFORE backends
  #                   initialise, so it never lists devices.
  #   --list-devices  loads the backends and enumerates GPUs, which is what
  #                   actually proves CUDA works and reveals where each backend
  #                   DLL was resolved from.
  # Using only --version produced a bogus "built with CUDA but no devices"
  # warning on a perfectly good build.
  Push-Location $bin
  try {
    $ver  = & $probe --version 2>&1 | ForEach-Object { [string]$_ }
    $devs = & $probe --list-devices 2>&1 | ForEach-Object { [string]$_ }
  } finally { Pop-Location }
  $out = @($ver) + @($devs)

  if (-not $out) {
    Write-Host "SMOKE TEST FAILED: no output at all." -ForegroundColor Red
    Write-Host "That is the missing-CUDA-runtime-DLL signature, not a build error." -ForegroundColor Yellow
    Write-Host "Put the CUDA bin\x64 directory on PATH, or copy cudart64_*/cublas*64_* next to the exe." -ForegroundColor Yellow
  } else {
    $ver | Where-Object { $_ -match 'version:|built with' } | ForEach-Object { Write-Host "  $_" }
    $nDev = ($out | Select-String 'Device \d+:|CUDA\d' | Measure-Object).Count
    if ($nDev -gt 0) { Write-Host "OK: $nDev CUDA device line(s) visible to the new build." -ForegroundColor Green }
    elseif (-not $CpuOnly) { Write-Host "WARNING: built with CUDA but no devices listed." -ForegroundColor Yellow }
    # Any backend resolved from outside this build dir means the test -- and any
    # benchmark run the same way -- is not measuring this build.
    $foreign = $out | Select-String 'load_backend' | Where-Object { $_.Line -notmatch [regex]::Escape($bin) }
    if ($foreign) {
      Write-Host "WARNING: backends loaded from OUTSIDE this build directory:" -ForegroundColor Yellow
      $foreign | ForEach-Object { Write-Host "  $($_.Line.Trim())" -ForegroundColor Yellow }
      Write-Host "Always run these binaries with their own bin\ as the working directory." -ForegroundColor Yellow
    }
  }
}

# The smoke test is diagnostic only. Do not let a probe's exit code mark a
# successful build as failed -- an invalid --version flag did exactly that.
$global:LASTEXITCODE = 0
$Error.Clear()

Write-Host "`nBinaries: $bin" -ForegroundColor Green
Write-Host "RUN THEM FROM HERE. Do not copy them over the baseline binaries beside the" -ForegroundColor Yellow
Write-Host "source tree -- those are immutable, and every recorded benchmark was produced" -ForegroundColor Yellow
Write-Host "by them. To compare, run BOTH and label which build produced which numbers." -ForegroundColor Yellow
Write-Host ""
Write-Host "  new build : $bin\llama-server.exe" -ForegroundColor DarkGray
Write-Host "  baseline  : $parentFull\llama-server.exe  (do not touch)" -ForegroundColor DarkGray






