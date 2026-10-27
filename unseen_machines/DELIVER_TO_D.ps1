# Administrative v3 delivery only. Run by Opus; never runs target analysis.
$ErrorActionPreference = 'Stop'
$sourceRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$destinationRoot = [System.IO.Path]::GetFullPath('D:\external_unseen')
if ($destinationRoot -ne 'D:\external_unseen') { throw 'Unexpected destination' }
if (-not (Test-Path -LiteralPath $destinationRoot -PathType Container)) { throw 'Existing destination missing' }
if ($sourceRoot -eq $destinationRoot) { throw 'Already delivered; do not recopy' }
$registration = Get-Content -LiteralPath (Join-Path $sourceRoot 'protocol.json') -Raw | ConvertFrom-Json
if ($registration.version -ne 3 -or $registration.amendment.number -ne 2) { throw 'Expected v3 / Amendment 2' }
if ($registration.output_root -ne 'D:\external_unseen\results_v3') { throw 'V3 output root differs' }

# Explicit top-level allowlist: never enumerate/copy destination data or results.
$topLevelNames = @('PROTOCOL.md', 'protocol.json', 'PROTOCOL_SHA256.txt',
    'PROTOCOL_v1.md', 'protocol_v1.json', 'PROTOCOL_SHA256_v1.txt', 'run_unseen_v1.py',
    'PROTOCOL_v2.md', 'protocol_v2.json', 'PROTOCOL_SHA256_v2.txt', 'run_unseen_v2.py',
    'STAGE1_VALIDATION_v2.md', 'validate_stage1_v2.py',
    'run_unseen.py', 'validate_stage1.py', 'STAGE1_VALIDATION.md', 'DELIVER_TO_D.ps1',
    'DEVIATIONS.md', 'STAGING_NOTICE.txt', 'source_snapshot_manifest.json', 'download_sha256.txt')
$mutableNames = @('PROTOCOL.md', 'protocol.json', 'PROTOCOL_SHA256.txt', 'run_unseen.py',
    'validate_stage1.py', 'STAGE1_VALIDATION.md', 'DELIVER_TO_D.ps1', 'DEVIATIONS.md')
$protocolNames = @('PROTOCOL.md', 'protocol.json', 'PROTOCOL_v1.md', 'protocol_v1.json',
    'PROTOCOL_v2.md', 'protocol_v2.json')
$seenNames = @()
foreach ($line in (Get-Content -LiteralPath (Join-Path $sourceRoot 'PROTOCOL_SHA256.txt'))) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    if ($line -notmatch '^([0-9a-f]{64})\s+\*(.+)$') { throw 'Invalid seal line' }
    $expectedHash = $Matches[1]
    $name = $Matches[2]
    if ($name -notin $protocolNames) { throw 'Unexpected protocol seal name' }
    $seenNames += $name
    if ((Get-FileHash -LiteralPath (Join-Path $sourceRoot $name) -Algorithm SHA256).Hash -ne $expectedHash) { throw "Broken protocol seal: $name" }
}
if ($seenNames.Count -ne 6 -or @($seenNames | Select-Object -Unique).Count -ne 6) { throw 'Expected six protocol seals' }

# Verify staged seals and existing frozen files; never merge/copy directories.
foreach ($property in $registration.frozen_files_sha256.PSObject.Properties) {
    $sealedPath = [System.IO.Path]::GetFullPath((Join-Path $sourceRoot $property.Name))
    if (-not $sealedPath.StartsWith($sourceRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Sealed path escaped staging' }
    if ((Get-FileHash -LiteralPath $sealedPath -Algorithm SHA256).Hash -ne $property.Value) { throw "Broken artifact seal: $($property.Name)" }
    if ($property.Name.StartsWith('frozen/')) {
        $target = Join-Path $destinationRoot $property.Name
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $property.Value) { throw "Existing frozen artifact differs or is absent: $($property.Name)" }
    } elseif ($property.Name -notin $topLevelNames) { throw 'Unallowlisted sealed top-level file' }
}
if ((Get-FileHash -LiteralPath (Join-Path $sourceRoot 'download_sha256.txt') -Algorithm SHA256).Hash -ne $registration.download_manifest_sha256) { throw 'Download manifest drift' }

function Get-ByteHash([byte[]] $Content) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash($Content))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}

# Hash all payloads before writing; immutable archive conflicts stop delivery.
$payloads = @()
$utf8 = [System.Text.UTF8Encoding]::new($false)
foreach ($name in $topLevelNames) {
    $source = Join-Path $sourceRoot $name
    $target = Join-Path $destinationRoot $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing artifact: $name" }
    if ((Test-Path -LiteralPath $target) -and -not (Test-Path -LiteralPath $target -PathType Leaf)) { throw "Target is not a file: $name" }
    $content = [System.IO.File]::ReadAllBytes($source)
    $sourceHash = Get-ByteHash $content
    $existingHash = $null
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        $existingHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($existingHash -ne $sourceHash -and $name -notin $mutableNames) { throw "Immutable destination conflict: $name" }
        if ($name -eq 'DEVIATIONS.md') {
            # Preserve v2 runtime STOP entries already on D:. This log is unsealed.
            $existingText = [System.IO.File]::ReadAllText($target, $utf8)
            $sourceLines = [System.IO.File]::ReadAllLines($source, $utf8)
            $existingLines = $existingText -split '\r?\n'
            $missingLines = @($sourceLines | Where-Object { $_ -and $_ -notin $existingLines })
            if ($missingLines.Count -gt 0) { $content = $utf8.GetBytes($existingText + "`n" + ($missingLines -join "`n") + "`n") }
            else { $content = [System.IO.File]::ReadAllBytes($target) }
        }
    }
    $payloads += [pscustomobject]@{ Name = $name; Target = $target; Content = $content;
        Hash = (Get-ByteHash $content); ExistingHash = $existingHash }
}
foreach ($payload in $payloads) {
    # Refuse changes since preflight; write only new/changed top-level files.
    $currentHash = $null
    if (Test-Path -LiteralPath $payload.Target -PathType Leaf) { $currentHash = (Get-FileHash -LiteralPath $payload.Target -Algorithm SHA256).Hash.ToLowerInvariant() }
    if ($currentHash -ne $payload.ExistingHash) { throw "Destination changed during delivery: $($payload.Name)" }
    if ($currentHash -ne $payload.Hash) { [System.IO.File]::WriteAllBytes($payload.Target, $payload.Content) }
    if ((Get-FileHash -LiteralPath $payload.Target -Algorithm SHA256).Hash -ne $payload.Hash) { throw "Delivery verification failed: $($payload.Name)" }
}
foreach ($property in $registration.frozen_files_sha256.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath (Join-Path $destinationRoot $property.Name) -Algorithm SHA256).Hash -ne $property.Value) { throw "Delivered seal failed: $($property.Name)" }
}
Write-Output 'V3 top-level delivery hash-verified; frozen tree retained; deviation history preserved. Data and results directories untouched. No analysis executed.'
