# Run by Opus after review. Stage 1 does not execute this script.
[CmdletBinding()]
param([switch]$VerifyOnly)
$ErrorActionPreference = 'Stop'
$stagingRoot = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$destinationRoot = [System.IO.Path]::GetFullPath('D:\external_unseen\comparators').TrimEnd('\')
$requiredStagingRoot = [System.IO.Path]::GetFullPath('C:\Users\Dell\AppData\Local\Temp\codex_unseen_comparators_20261027').TrimEnd('\')
if ($stagingRoot -ne $requiredStagingRoot) { throw 'Run delivery only from the registered staging folder.' }
if ($destinationRoot -ne 'D:\external_unseen\comparators') { throw 'Unexpected destination.' }

$sealNames = @('COMPARATORS_PROTOCOL.md', 'comparators_protocol.json')
$sealed = @{}
foreach ($line in Get-Content -LiteralPath (Join-Path $stagingRoot 'COMPARATORS_SHA256.txt')) {
    if ($line.Trim()) {
        $parts = $line -split '\s+', 2
        $name = $parts[1].TrimStart(' ', '*')
        if ($sealed.ContainsKey($name)) { throw "Duplicate seal: $name" }
        $sealed[$name] = $parts[0]
    }
}
if ($sealed.Count -ne 2) { throw 'Expected exactly two protocol seals.' }
foreach ($name in $sealNames) {
    if (-not $sealed.ContainsKey($name)) { throw "Missing protocol seal: $name" }
    $digest = (Get-FileHash -LiteralPath (Join-Path $stagingRoot $name) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($digest -ne $sealed[$name]) { throw "Protocol seal broken: $name" }
}
$protocol = Get-Content -LiteralPath (Join-Path $stagingRoot 'comparators_protocol.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($protocol.status -ne 'SEALED_STAGE1_COMPLETE') { throw 'Stage 1 incomplete.' }
$files = @{}
foreach ($property in $protocol.bundle_files_sha256.PSObject.Properties) { $files[$property.Name] = $property.Value }
foreach ($name in $sealNames) { $files[$name] = $sealed[$name] }
$files['COMPARATORS_SHA256.txt'] = (Get-FileHash -LiteralPath (Join-Path $stagingRoot 'COMPARATORS_SHA256.txt') -Algorithm SHA256).Hash.ToLowerInvariant()

# Verify all paths and all source hashes BEFORE creating the destination.
foreach ($relative in $files.Keys) {
    if ([System.IO.Path]::IsPathRooted($relative)) { throw "Absolute artifact path: $relative" }
    $sourcePath = [System.IO.Path]::GetFullPath((Join-Path $stagingRoot $relative))
    $destinationPath = [System.IO.Path]::GetFullPath((Join-Path $destinationRoot $relative))
    if (-not $sourcePath.StartsWith($stagingRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Source escaped staging.' }
    if (-not $destinationPath.StartsWith($destinationRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Destination escaped comparators.' }
    if ($relative -match '(^|[\\/])(results|scores|\.git|_codex|__pycache__)([\\/]|$)') { throw "Prohibited delivery artifact: $relative" }
    if ((Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $files[$relative]) { throw "Staging artifact drift: $relative" }
}
if ($VerifyOnly) { Write-Output "PASS: $($files.Count) delivery artifacts verified; nothing copied."; return }

# Never merge into, overwrite, remove or rename an existing destination.
if (Test-Path -LiteralPath $destinationRoot) { throw 'Destination already exists. Review it manually; delivery never overwrites.' }
if (-not (Test-Path -LiteralPath 'D:\external_unseen' -PathType Container)) { throw 'Missing external evaluation root.' }
New-Item -ItemType Directory -Path $destinationRoot | Out-Null
foreach ($relative in ($files.Keys | Sort-Object)) {
    $sourcePath = [System.IO.Path]::GetFullPath((Join-Path $stagingRoot $relative))
    $destinationPath = [System.IO.Path]::GetFullPath((Join-Path $destinationRoot $relative))
    $parentPath = [System.IO.Path]::GetDirectoryName($destinationPath)
    if (-not $parentPath.StartsWith($destinationRoot, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Parent escaped destination.' }
    if (-not (Test-Path -LiteralPath $parentPath)) { New-Item -ItemType Directory -Path $parentPath -Force | Out-Null }
    Copy-Item -LiteralPath $sourcePath -Destination $destinationPath
    if ((Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $files[$relative]) { throw "Copied hash mismatch: $relative" }
}
Write-Output "Delivered and verified $($files.Count) files to $destinationRoot."
Write-Output 'Stage 2: Set-Location -LiteralPath D:\external_unseen\comparators'
Write-Output 'python -I -B .\run_comparators.py --protocol .\comparators_protocol.json --run'
