<#
.SYNOPSIS
    Builds MIDIMusic for Windows: a PyInstaller onedir bundle and an installer.

.DESCRIPTION
    Run from a clean checkout on Windows with Python 3.12 on PATH. Torch is not
    part of the build; the app installs it on first run to match the GPU.

.PARAMETER SkipInstaller
    Build the application folder but skip Inno Setup.

.EXAMPLE
    .\build.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipInstaller,
    [string]$Python = "py -3.12"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $root

try {
    Write-Host "== Checking Python ==" -ForegroundColor Cyan
    $version = & cmd /c "$Python -c ""import sys;print('.'.join(map(str,sys.version_info[:2])))"""
    Write-Host "Python $version"
    if ($version -ne "3.12") {
        # AMD's Windows ROCm wheels are cp312-only, and tinysoundfont publishes
        # no cp313 wheels. Building on anything else produces a bundle that
        # cannot later install a GPU runtime.
        Write-Warning "Python 3.12 is expected. GPU runtime installation may fail on $version."
    }

    Write-Host "== Creating build environment ==" -ForegroundColor Cyan
    $venv = Join-Path $root ".venv-build"
    if (-not (Test-Path $venv)) { & cmd /c "$Python -m venv `"$venv`"" }
    $py = Join-Path $venv "Scripts\python.exe"

    & $py -m pip install --upgrade pip wheel | Out-Null
    & $py -m pip install -e . | Out-Null
    & $py -m pip install pyinstaller | Out-Null

    Write-Host "== Cleaning previous build ==" -ForegroundColor Cyan
    foreach ($dir in @("build", "dist")) {
        if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    }

    Write-Host "== Running PyInstaller ==" -ForegroundColor Cyan
    & $py -m PyInstaller --noconfirm --clean "packaging\windows\midimusic.spec"
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

    $exe = Join-Path $root "dist\MIDIMusic\MIDIMusic.exe"
    if (-not (Test-Path $exe)) { throw "Build did not produce $exe" }

    $size = (Get-ChildItem -Recurse "dist\MIDIMusic" | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host ("Bundle size: {0:N0} MB" -f $size) -ForegroundColor Green

    Write-Host "== Verifying the bundle ==" -ForegroundColor Cyan
    # These checks used to live in CI. They run here instead, because building
    # on a GitHub windows-latest runner costs 2x minutes and the build is not
    # something every push needs to prove.

    # The worker is executed by a different interpreter, so it has to survive
    # as a real .py file rather than only as bytecode inside the archive.
    $worker = "dist\MIDIMusic\_internal\midimusic\worker\runner.py"
    if (-not (Test-Path $worker)) { throw "worker script was not bundled: $worker" }

    # uv builds the compute runtime. Without it the installed app cannot
    # provision anything, because it has no interpreter of its own.
    $uv = "dist\MIDIMusic\_internal\uv\uv.exe"
    if (-not (Test-Path $uv)) { throw "uv was not bundled - the app would not be self-contained" }

    # A bundle this large means torch has been pulled in, which defeats the
    # whole point of provisioning it per-GPU after install.
    if ($size -gt 900) { throw ("bundle is {0:N0} MB - is torch being pulled in?" -f $size) }

    # Catch a missing hidden import before a user does.
    & $py -c "import midimusic, midimusic.ui.main_window, midimusic.core.service; print('imports OK')"
    if ($LASTEXITCODE -ne 0) { throw "the bundled package failed to import" }

    Write-Host "Bundle checks passed" -ForegroundColor Green

    if (-not $SkipInstaller) {
        Write-Host "== Building installer ==" -ForegroundColor Cyan
        $iscc = Get-Command "iscc.exe" -ErrorAction SilentlyContinue
        if (-not $iscc) {
            $candidates = @(
                "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
                "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
            )
            $iscc = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        }
        if (-not $iscc) {
            Write-Warning "Inno Setup not found; skipping installer. Install it from https://jrsoftware.org/isdl.php"
        } else {
            & $iscc "packaging\windows\installer.iss"
            if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE" }
            Write-Host "Installer written to packaging\windows\Output" -ForegroundColor Green
        }
    }

    Write-Host "`nDone." -ForegroundColor Green
    Write-Host "Unsigned builds trigger SmartScreen until they earn reputation."
    Write-Host "Signing helps, but note that since 2024 EV certificates no longer"
    Write-Host "grant immediate reputation -- OV and EV now build it the same way."
}
finally {
    Pop-Location
}
