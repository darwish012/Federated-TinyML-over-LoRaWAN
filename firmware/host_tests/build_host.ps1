# Builds the host test library (tfcore.dll, Windows x86_64) from the portable core
# with `zig c++` and checks nothing else.  Needs zig (PyPI "ziglang").
#
#   * If `python -m ziglang` works on this machine it is used directly.
#   * Otherwise pass -Zig <path-to-zig-executable> (for example the WSL copy:
#     -Zig "wsl:/tmp/zg/ziglang/zig", which runs zig inside WSL and cross-compiles
#     to a Windows DLL that the Windows Python loads with ctypes).
param(
    [string]$Zig = "python",
    [string]$Python = "python",
    [string]$Out = "$PSScriptRoot\tfcore.dll",
    [switch]$V4
)
$ErrorActionPreference = "Stop"
$env:ZIG_GLOBAL_CACHE_DIR = Join-Path $PSScriptRoot '.zig-cache'
$env:ZIG_LOCAL_CACHE_DIR = Join-Path $PSScriptRoot '.zig-local-cache'
$core = Join-Path $PSScriptRoot "..\common\tinyfl_core"
$srcs = @("tf_common","tf_fft","tf_features","tf_rawfile","tf_model","tf_alarm","tf_protocol","tf_machine") | ForEach-Object { Join-Path $core "$_.cpp" }
$srcs += Join-Path $PSScriptRoot "tf_capi.cpp"
$srcs += Join-Path $PSScriptRoot "v4_capi.cpp"
$flags = @("c++", "-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math", "-shared", "-target", "x86_64-windows-gnu", "-Wall", "-Wextra", "-Wno-unused-parameter")
$flags += @("-I", (Join-Path $PSScriptRoot "mock"))
if ($V4) { $flags += "-DTINYFL_SD_V4=1" }

function ToWsl([string]$p) { $p = (Resolve-Path -LiteralPath (Split-Path $p -Parent)).Path + "\" + (Split-Path $p -Leaf); return "/mnt/" + $p.Substring(0,1).ToLower() + $p.Substring(2).Replace("\","/") }

$ErrorActionPreference = "Continue"   # zig prints libunwind warnings on stderr
if ($Zig -like "wsl:*") {
    $zigPath = $Zig.Substring(4)
    $wslFlags = $flags | ForEach-Object { if ($_ -eq (Join-Path $PSScriptRoot "mock")) { ToWsl $_ } else { $_ } }
    $args2 = $wslFlags + ($srcs | ForEach-Object { ToWsl $_ }) + @("-o", (ToWsl $Out))
    & wsl -e $zigPath @args2
} elseif ($Zig -eq "python") {
    & $Python -m ziglang @flags @srcs -o $Out
} else {
    & $Zig @flags @srcs -o $Out
}
if ($LASTEXITCODE -ne 0) { throw "host build failed" }
Write-Host "built $Out"
