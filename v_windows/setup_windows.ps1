param(
    [string]$EnvName = "3dgs_rtmaterial_windows",
    [switch]$SkipOptix,
    [switch]$ForceDownload
)

$ErrorActionPreference = "Stop"
$PortRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VendorRoot = Join-Path $PortRoot "vendor"
$DownloadRoot = Join-Path $VendorRoot "downloads"

function Stop-WithHelp([string]$Message) {
    Write-Host "`n$Message" -ForegroundColor Red
    exit 1
}

function Assert-LastExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        Stop-WithHelp "$Step failed with exit code $LASTEXITCODE."
    }
}

function Enable-MsvcEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $VsWhere)) {
        Stop-WithHelp "Visual Studio 2022 Build Tools was not found. Install 'Desktop development with C++' and the Windows 10/11 SDK."
    }
    $VsRoot = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $VsRoot) {
        Stop-WithHelp "MSVC x64 tools were not found. Add the 'Desktop development with C++' workload."
    }
    $VsDevCmd = Join-Path $VsRoot "Common7\Tools\VsDevCmd.bat"
    cmd.exe /s /c "`"$VsDevCmd`" -arch=x64 -host_arch=x64 >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        Stop-WithHelp "VsDevCmd completed, but cl.exe is still unavailable."
    }
}

function Expand-GitHubArchive([string]$Url, [string]$Destination, [string]$ArchiveName) {
    $Zip = Join-Path $DownloadRoot $ArchiveName
    $Scratch = Join-Path $DownloadRoot ([IO.Path]::GetFileNameWithoutExtension($ArchiveName) + "-expanded")
    if ($ForceDownload -or -not (Test-Path $Zip)) {
        Write-Host "Downloading $Url"
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Zip
    }
    if (Test-Path $Destination) { Remove-Item -LiteralPath $Destination -Recurse -Force }
    if (Test-Path $Scratch) { Remove-Item -LiteralPath $Scratch -Recurse -Force }
    Expand-Archive -LiteralPath $Zip -DestinationPath $Scratch -Force
    $Top = Get-ChildItem -LiteralPath $Scratch -Directory | Select-Object -First 1
    if (-not $Top) { Stop-WithHelp "Archive $Zip did not contain a top-level directory." }
    $Parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $Parent -Force | Out-Null
    Move-Item -LiteralPath $Top.FullName -Destination $Destination
    Remove-Item -LiteralPath $Scratch -Recurse -Force
}

if (-not (Get-Command conda.exe -ErrorAction SilentlyContinue)) {
    Stop-WithHelp "Conda was not found. Install Miniconda/Anaconda and reopen PowerShell."
}
Enable-MsvcEnvironment

New-Item -ItemType Directory -Path $DownloadRoot -Force | Out-Null
$Existing = (& conda.exe env list --json | ConvertFrom-Json).envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } | Select-Object -First 1
if (-not $Existing) {
    & conda.exe create -n $EnvName python=3.11 pip -y
    Assert-LastExit "Creating Conda environment"
    $Existing = (& conda.exe env list --json | ConvertFrom-Json).envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } | Select-Object -First 1
}
$EnvPython = Join-Path $Existing "python.exe"

$Nvcc = Get-ChildItem -LiteralPath $Existing -Filter nvcc.exe -File -Recurse | Select-Object -First 1
if (-not $Nvcc) {
    Write-Host "Installing CUDA Toolkit 12.8 inside the Conda environment..."
    & conda.exe install -n $EnvName -y -c nvidia/label/cuda-12.8.0 cuda-toolkit=12.8
    Assert-LastExit "Installing CUDA Toolkit"
    $Nvcc = Get-ChildItem -LiteralPath $Existing -Filter nvcc.exe -File -Recurse | Select-Object -First 1
}
if (-not $Nvcc) { Stop-WithHelp "CUDA installation completed, but nvcc.exe was not found under $Existing." }
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PortRoot "fix_conda_activation.ps1") -EnvPath $Existing
Assert-LastExit "Disabling incompatible Conda compiler activation hooks"
$env:CUDA_HOME = Split-Path -Parent (Split-Path -Parent $Nvcc.FullName)
$env:CUDA_PATH = $env:CUDA_HOME
$env:PATH = "$($Nvcc.Directory.FullName);$env:PATH"
$env:LIB = "$(Join-Path $Existing 'Library\lib');$env:LIB"
$env:INCLUDE = "$(Join-Path $Existing 'Library\include\targets\x64');$(Join-Path $Existing 'Library\include');$env:INCLUDE"
$NvccText = (& $Nvcc.FullName --version) -join "`n"
if ($NvccText -notmatch 'release 12\.8') {
    Stop-WithHelp "CUDA 12.8 was requested, but nvcc reports:`n$NvccText"
}

Write-Host "Installing PyTorch CUDA 12.8 runtime..."
& $EnvPython -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
Assert-LastExit "Installing PyTorch"
& $EnvPython -m pip install -r (Join-Path $PortRoot "requirements-windows.txt")
Assert-LastExit "Installing Python dependencies"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PortRoot "install_sam2.ps1") -EnvPath $Existing -ModelSize large
Assert-LastExit "Installing SAM 2.1"

Expand-GitHubArchive "https://github.com/Jumpat/SegAnyGAussians/archive/refs/heads/main.zip" (Join-Path $VendorRoot "saga") "saga-main.zip"
$env:TORCH_CUDA_ARCH_LIST = "12.0"

$Extensions = @(
    "diff-gaussian-rasterization",
    "diff-gaussian-rasterization_contrastive_f",
    "simple-knn"
)
foreach ($Extension in $Extensions) {
    $Source = Join-Path $VendorRoot "saga\submodules\$Extension"
    $ModuleName = $Extension.Replace('-', '_')
    if (-not $ForceDownload) {
        & $EnvPython -c "import torch, importlib; importlib.import_module('$ModuleName')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "$Extension is already importable; skipping rebuild."
            continue
        }
    }
    $SetupPy = Join-Path $Source "setup.py"
    $SetupText = Get-Content -LiteralPath $SetupPy -Raw
    if ($SetupText -notmatch 'allow-unsupported-compiler') {
        # CUDA 12.8 rejects the newer MSVC shipped with VS 2026 before compiling.
        # NVIDIA documents this nvcc switch for overriding only that version gate.
        $SetupText = $SetupText.Replace('"nvcc": [', '"nvcc": ["-allow-unsupported-compiler", ')
        Set-Content -LiteralPath $SetupPy -Value $SetupText -Encoding UTF8
    }
    # Avoid feeding PyTorch's Python/Dynamo binding headers through nvcc.
    # New MSVC versions see an ambiguous std namespace there; CUDA sources
    # only need the lightweight tensor type declarations.
    Get-ChildItem -LiteralPath $Source -Recurse -File -Include *.cu,*.h | ForEach-Object {
        $CudaText = Get-Content -LiteralPath $_.FullName -Raw
        if ($null -ne $CudaText -and $CudaText.Contains('#include <torch/extension.h>')) {
            $CudaText = $CudaText.Replace('#include <torch/extension.h>', '#include <torch/types.h>')
            Set-Content -LiteralPath $_.FullName -Value $CudaText -Encoding UTF8
        }
    }
    Get-ChildItem -LiteralPath $Source -Recurse -File -Filter *.cpp | ForEach-Object {
        $CppText = Get-Content -LiteralPath $_.FullName -Raw
        if ($null -ne $CppText -and $CppText.Contains('#include <torch/types.h>')) {
            $CppText = $CppText.Replace('#include <torch/types.h>', '#include <torch/extension.h>')
            Set-Content -LiteralPath $_.FullName -Value $CppText -Encoding UTF8
        }
    }
    Write-Host "Building $Extension..."
    & $EnvPython -m pip install $Source --no-build-isolation
    Assert-LastExit "Building $Extension"
}

if (-not $SkipOptix) {
    Write-Host "Preparing NVIDIA 3DGRUT/OptiX source without Git..."
    $GrutRoot = Join-Path $VendorRoot "3dgrut"
    Expand-GitHubArchive "https://github.com/nv-tlabs/3dgrut/archive/refs/heads/main.zip" $GrutRoot "3dgrut-main.zip"
    Expand-GitHubArchive "https://github.com/NVlabs/tiny-cuda-nn/archive/refs/heads/master.zip" (Join-Path $GrutRoot "thirdparty\tiny-cuda-nn") "tiny-cuda-nn-master.zip"
    Expand-GitHubArchive "https://github.com/fmtlib/fmt/archive/refs/heads/master.zip" (Join-Path $GrutRoot "thirdparty\tiny-cuda-nn\dependencies\fmt") "fmt-master.zip"
    Expand-GitHubArchive "https://github.com/NVIDIA/optix-dev/archive/refs/tags/v7.5.0.zip" (Join-Path $GrutRoot "threedgrt_tracer\dependencies\optix-dev") "optix-dev-v7.5.0.zip"
    $JitPy = Join-Path $GrutRoot "threedgrut\utils\jit.py"
    $JitText = Get-Content -LiteralPath $JitPy -Raw
    if ($JitText -notmatch 'slang_executable =') {
        $JitText = $JitText.Replace(
            'slang_build_env["PATH"] += ";" if os.name == "nt" else ":"',
            'slang_build_env["PATH"] += ";" if os.name == "nt" else ":"' + "`n    " + 'slang_executable = "slangc"'
        )
        $JitText = $JitText.Replace(
            'slang_mod = importlib.import_module("slangtorch")',
            'slang_mod = importlib.import_module("slangtorch")' + "`n        " + 'if os.name == "nt":' + "`n            " + 'slang_executable = os.path.join(os.path.dirname(slang_mod.__file__), "bin", "slangc.exe")'
        )
        $JitText = $JitText.Replace('            "slangc",', '            slang_executable,')
        Set-Content -LiteralPath $JitPy -Value $JitText -Encoding UTF8
    }
    $JitText = Get-Content -LiteralPath $JitPy -Raw
    if ($JitText -notmatch 'cuda_cflags\.append\("-allow-unsupported-compiler"\)') {
        $JitText = $JitText.Replace(
            '    if extra_cuda_cflags is not None:',
            '    if os.name == "nt":' + "`n        " + 'cuda_cflags.remove("-Xcompiler=-fno-strict-aliasing")' + "`n        " + 'cuda_cflags.append("-allow-unsupported-compiler")' + "`n    " + 'if extra_cuda_cflags is not None:'
        )
        Set-Content -LiteralPath $JitPy -Value $JitText -Encoding UTF8
    }
    Get-ChildItem -LiteralPath (Join-Path $GrutRoot "threedgrt_tracer") -Recurse -File -Include *.cu,*.h | ForEach-Object {
        $TracerText = Get-Content -LiteralPath $_.FullName -Raw
        if ($null -ne $TracerText -and $TracerText.Contains('#include <torch/extension.h>')) {
            $TracerText = $TracerText.Replace('#include <torch/extension.h>', '#include <torch/types.h>')
            Set-Content -LiteralPath $_.FullName -Value $TracerText -Encoding UTF8
        }
    }
    # The PowerShell -Include filter can also match bindings.cpp during a recursive walk.
    # This translation unit owns PYBIND11_MODULE and must retain torch/extension.h.
    $TracerBindings = Join-Path $GrutRoot "threedgrt_tracer\bindings.cpp"
    $BindingsText = Get-Content -LiteralPath $TracerBindings -Raw
    if ($null -ne $BindingsText -and $BindingsText.Contains('#include <torch/types.h>')) {
        $BindingsText = $BindingsText.Replace('#include <torch/types.h>', '#include <torch/extension.h>')
        Set-Content -LiteralPath $TracerBindings -Value $BindingsText -Encoding UTF8
    }
    & $EnvPython -m pip install -e $GrutRoot --no-build-isolation
    Assert-LastExit "Installing 3DGRUT"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PortRoot "build_optix.ps1") -EnvName $EnvName
    Assert-LastExit "Building the 3DGRUT OptiX plugin"
}

Write-Host "`nRunning diagnostics..."
& $EnvPython (Join-Path $PortRoot "diagnose.py")
Assert-LastExit "Diagnostics"
Write-Host "`nInstallation complete. Run:" -ForegroundColor Green
Write-Host "conda activate $EnvName"
Write-Host "python `"$PortRoot\launcher.py`" viewer -m C:\path\to\scene"
