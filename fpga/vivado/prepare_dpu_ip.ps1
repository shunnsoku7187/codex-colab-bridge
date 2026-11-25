$ErrorActionPreference = 'Stop'
$source = 'C:\AMDDesignTools\DPUCZDX8G_ip_repo_VAI_v3.0'
$target = 'C:\AMDDesignTools\DPUCZDX8G_ip_repo_VAI_v3.0_ultra96_patch'
$relativeTcl = 'DPUCZDX8G_v4_1_0\bd\bd.tcl'
$sourceTcl = Join-Path $source $relativeTcl
$targetTcl = Join-Path $target $relativeTcl

if (-not (Test-Path -LiteralPath $sourceTcl)) { throw "Missing official DPU IP: $sourceTcl" }
if (-not (Test-Path -LiteralPath $target)) {
    Copy-Item -LiteralPath $source -Destination $target -Recurse
}

$old = 'set_property "CONFIG.ARCH"                    {144}         $ip'
$new = 'set_property "CONFIG.ARCH"                    {512}         $ip'
$content = [IO.File]::ReadAllText($targetTcl)
if ($content.Contains($old)) {
    if ($content.Split([string[]]@($old), [StringSplitOptions]::None).Length -ne 2) {
        throw 'Expected exactly one ARCH default to patch'
    }
    [IO.File]::WriteAllText($targetTcl, $content.Replace($old, $new))
} elseif (-not $content.Contains($new)) {
    throw 'DPU IP init Tcl differs from the expected release'
}

Write-Output "Original SHA256=$((Get-FileHash -LiteralPath $sourceTcl -Algorithm SHA256).Hash)"
Write-Output "Patched SHA256=$((Get-FileHash -LiteralPath $targetTcl -Algorithm SHA256).Hash)"
Write-Output "Prepared DPU IP at $target"
