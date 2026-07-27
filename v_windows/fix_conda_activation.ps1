param(
    [string]$EnvName = "3dgs_rtmaterial_windows",
    [string]$EnvPath
)

$ErrorActionPreference = "Stop"

if (-not $EnvPath) {
    if (-not (Get-Command conda.exe -ErrorAction SilentlyContinue)) {
        throw "Conda was not found. Run this script from Anaconda Prompt or pass -EnvPath."
    }
    $EnvPath = (& conda.exe env list --json | ConvertFrom-Json).envs |
        Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } |
        Select-Object -First 1
}
if (-not $EnvPath -or -not (Test-Path -LiteralPath $EnvPath)) {
    throw "Conda environment '$EnvName' was not found."
}

# NVIDIA's CUDA 12.8 Conda metapackage currently pulls in a VS2017 helper that
# hard-codes MSVC 14.16. It cannot initialize a VS 2026 installation. The nvcc
# hook also appends \targets\x64 before LIBRARY_INC is guaranteed to exist.
# Our build scripts locate modern MSVC and configure CUDA paths themselves.
$Hooks = @(
    "etc\conda\activate.d\vs2017_compiler_vars.bat",
    "etc\conda\activate.d\vs2017_get_vsinstall_dir.bat",
    "etc\conda\activate.d\~cuda-nvcc_activate.bat",
    "etc\conda\deactivate.d\~cuda-nvcc_deactivate.bat"
)

foreach ($RelativePath in $Hooks) {
    $Source = Join-Path $EnvPath $RelativePath
    $Disabled = "$Source.disabled"
    if (Test-Path -LiteralPath $Source) {
        Move-Item -LiteralPath $Source -Destination $Disabled -Force
        Write-Host "Disabled incompatible Conda hook: $RelativePath"
    }
}

Write-Host "Conda activation hooks are compatible with the Windows launcher." -ForegroundColor Green
Write-Host "Open a new terminal before activating '$EnvName' again."
