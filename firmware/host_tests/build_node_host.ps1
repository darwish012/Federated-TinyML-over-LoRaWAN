param([string]$Python="python")
$ErrorActionPreference="Stop"
$env:ZIG_GLOBAL_CACHE_DIR=Join-Path $PSScriptRoot '.zig-cache'
$env:ZIG_LOCAL_CACHE_DIR=Join-Path $PSScriptRoot '.zig-local-cache'
$mock=Join-Path $PSScriptRoot 'mock'
$source=Join-Path $PSScriptRoot 'node_v4_capi.cpp'
$output=Join-Path $PSScriptRoot 'tfnode_v4.dll'
& $Python -m ziglang c++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math -shared -target x86_64-windows-gnu -I $mock $source -o $output
if($LASTEXITCODE -ne 0) { throw 'host node build failed' }
Write-Host "built $output"
