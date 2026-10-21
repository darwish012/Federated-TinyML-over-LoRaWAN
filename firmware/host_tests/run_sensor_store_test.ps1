# Builds and runs test_sensor_store.cpp (the sensor-only build's feature store) with zig c++.
param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$env:ZIG_GLOBAL_CACHE_DIR = Join-Path $PSScriptRoot '.zig-cache'
$env:ZIG_LOCAL_CACHE_DIR = Join-Path $PSScriptRoot '.zig-local-cache'
$core = Join-Path $PSScriptRoot "..\common\tinyfl_core"
$srcs = @("tf_common","tf_fft","tf_features","tf_rawfile","tf_model","tf_alarm","tf_protocol","tf_machine") | ForEach-Object { Join-Path $core "$_.cpp" }
$exe = Join-Path $PSScriptRoot "test_sensor_store.exe"
$ErrorActionPreference = "Continue"   # zig prints libunwind warnings on stderr
& $Python -m ziglang c++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math -target x86_64-windows-gnu -DTINYFL_SD_V4=1 `
    -I (Join-Path $PSScriptRoot "mock") (Join-Path $PSScriptRoot "test_sensor_store.cpp") @srcs -o $exe
if ($LASTEXITCODE -ne 0) { throw "sensor store test build failed" }
& $exe
if ($LASTEXITCODE -ne 0) { throw "sensor store test failed" }
