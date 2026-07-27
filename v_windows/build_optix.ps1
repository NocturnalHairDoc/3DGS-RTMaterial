param([string]$EnvName = "3dgs_rtmaterial_windows")

$ErrorActionPreference = "Stop"
$PortRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvPath = (& conda.exe env list --json | ConvertFrom-Json).envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } | Select-Object -First 1
if (-not $EnvPath) { throw "Conda environment '$EnvName' was not found." }

$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$VsRoot = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VsRoot) { throw "Visual Studio C++ x64 tools were not found." }
$VsDevCmd = Join-Path $VsRoot "Common7\Tools\VsDevCmd.bat"
cmd.exe /s /c "`"$VsDevCmd`" -arch=x64 -host_arch=x64 >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
}

$CudaRoot = Join-Path $EnvPath "Library"
$env:CUDA_HOME = $CudaRoot
$env:CUDA_PATH = $CudaRoot
$env:PATH = "$(Join-Path $CudaRoot 'bin');$env:PATH"
$env:LIB = "$(Join-Path $CudaRoot 'lib');$env:LIB"
$env:INCLUDE = "$(Join-Path $CudaRoot 'include\targets\x64');$(Join-Path $CudaRoot 'include');$env:INCLUDE"
$env:TORCH_CUDA_ARCH_LIST = "12.0"

Push-Location $PortRoot
try {
    & (Join-Path $EnvPath "python.exe") -c "from optix_integration import build_3dgrt_plugin; raise SystemExit(0 if build_3dgrt_plugin(verbose=True) else 1)"
    $BuildExit = $LASTEXITCODE
    if ($BuildExit -eq 0) {
        $BuiltPlugin = Get-ChildItem (Join-Path $env:LOCALAPPDATA "torch_extensions") -Filter "lib3dgrt_cc.pyd" -File -Recurse |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $BuiltPlugin) { throw "Compilation succeeded, but lib3dgrt_cc.pyd was not found in the PyTorch cache." }
        $PackageDir = Join-Path $PortRoot "vendor\3dgrut\threedgrt_tracer"
        Copy-Item -LiteralPath $BuiltPlugin.FullName -Destination (Join-Path $PackageDir "lib3dgrt_cc.pyd") -Force
        Write-Host "Installed OptiX plugin into $PackageDir" -ForegroundColor Green
    }
    exit $BuildExit
} finally {
    Pop-Location
}
