param(
    [string]$EnvName = "3dgs_rtmaterial_windows",
    [string]$EnvPath,
    [ValidateSet("large", "base_plus", "small", "tiny")]
    [string]$ModelSize = "large",
    [switch]$SkipCheckpoint
)

$ErrorActionPreference = "Stop"
$PortRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VendorRoot = Join-Path $PortRoot "vendor"
$DownloadRoot = Join-Path $VendorRoot "downloads"
$Sam2Root = Join-Path $VendorRoot "sam2"

if (-not $EnvPath) {
    $EnvPath = (& conda.exe env list --json | ConvertFrom-Json).envs |
        Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } |
        Select-Object -First 1
}
if (-not $EnvPath) { throw "Conda environment '$EnvName' was not found." }
$Python = Join-Path $EnvPath "python.exe"

New-Item -ItemType Directory -Path $DownloadRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $Sam2Root)) {
    $Archive = Join-Path $DownloadRoot "sam2-main-full.zip"
    $Expanded = Join-Path $DownloadRoot "sam2-expanded-install"
    if (-not (Test-Path -LiteralPath $Archive)) {
        Write-Host "Downloading Meta SAM 2 source archive (no Git)..."
        & curl.exe -L --fail --output $Archive "https://codeload.github.com/facebookresearch/sam2/zip/refs/heads/main"
        if ($LASTEXITCODE -ne 0) { throw "Downloading SAM 2 source failed." }
    }
    Expand-Archive -LiteralPath $Archive -DestinationPath $Expanded -Force
    Move-Item -LiteralPath (Join-Path $Expanded "sam2-main") -Destination $Sam2Root
}

Write-Host "Installing SAM 2 without its optional CUDA morphology extension..."
$PreviousBuildCuda = $env:SAM2_BUILD_CUDA
$env:SAM2_BUILD_CUDA = "0"
try {
    & $Python -m pip install -e $Sam2Root --no-build-isolation
    if ($LASTEXITCODE -ne 0) { throw "Installing SAM 2 failed." }
}
finally {
    $env:SAM2_BUILD_CUDA = $PreviousBuildCuda
}

if (-not $SkipCheckpoint) {
    $Models = @{
        large     = @{ File="sam2.1_hiera_large.pt";     Url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" }
        base_plus = @{ File="sam2.1_hiera_base_plus.pt"; Url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt" }
        small     = @{ File="sam2.1_hiera_small.pt";     Url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" }
        tiny      = @{ File="sam2.1_hiera_tiny.pt";      Url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" }
    }
    $CheckpointRoot = Join-Path $PortRoot "checkpoints"
    New-Item -ItemType Directory -Path $CheckpointRoot -Force | Out-Null
    $Model = $Models[$ModelSize]
    $Checkpoint = Join-Path $CheckpointRoot $Model.File
    if (-not (Test-Path -LiteralPath $Checkpoint)) {
        Write-Host "Downloading SAM 2.1 $ModelSize checkpoint..."
        & curl.exe -L --fail --output $Checkpoint $Model.Url
        if ($LASTEXITCODE -ne 0) { throw "Downloading SAM 2.1 checkpoint failed." }
    }
    Write-Host "Checkpoint: $Checkpoint"
}

& $Python -c "import sam2; from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator; print('SAM2_IMPORT_OK', sam2.__path__[0])"
if ($LASTEXITCODE -ne 0) { throw "SAM 2 import validation failed." }
