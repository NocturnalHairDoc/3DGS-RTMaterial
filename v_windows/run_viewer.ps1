param(
    [Parameter(Mandatory=$true)][string]$ModelPath,
    [int]$FeatureIteration = 10000,
    [int]$SceneIteration = 30000,
    [double]$Scale = 2.0,
    [string]$EnvName = "3dgs_rtmaterial_windows"
)

$ErrorActionPreference = "Stop"
$PortRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvPath = (& conda.exe env list --json | ConvertFrom-Json).envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } | Select-Object -First 1
if (-not $EnvPath) { throw "Conda environment '$EnvName' was not found. Run setup_windows.ps1 first." }
& (Join-Path $EnvPath "python.exe") (Join-Path $PortRoot "launcher.py") viewer `
    -m $ModelPath -f $FeatureIteration -s $SceneIteration --scale $Scale
exit $LASTEXITCODE
