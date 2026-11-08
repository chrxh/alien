# Bootstraps the vcpkg tool pinned by the submodule. The release download from github.com fails
# now and then (HTTP 504) while the GitHub API still serves the same asset.
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$vcpkgRoot = (Resolve-Path "$PSScriptRoot/../../external/vcpkg").Path
$tag = (ConvertFrom-StringData (Get-Content "$vcpkgRoot/scripts/vcpkg-tool-metadata.txt" -Raw)).VCPKG_TOOL_RELEASE_TAG
$tool = Join-Path $vcpkgRoot 'vcpkg.exe'

function Test-PinnedVersion([string]$path)
{
    try
    {
        (Test-Path $path) -and [bool]((& $path version --disable-metrics) -match "version $tag-")
    }
    catch
    {
        $false
    }
}

function Get-ThroughApi
{
    $headers = @{}
    if ($env:GITHUB_TOKEN)
    {
        $headers.Authorization = "Bearer $env:GITHUB_TOKEN"
    }
    $release = Invoke-RestMethod "https://api.github.com/repos/microsoft/vcpkg-tool/releases/tags/$tag" -Headers $headers -MaximumRetryCount 5 -RetryIntervalSec 5
    $asset = $release.assets | Where-Object name -EQ 'vcpkg.exe'
    $headers.Accept = 'application/octet-stream'
    $part = Join-Path $vcpkgRoot 'vcpkg.download.exe'
    Invoke-WebRequest $asset.url -Headers $headers -OutFile $part -MaximumRetryCount 5 -RetryIntervalSec 5
    $signature = Get-AuthenticodeSignature $part
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation')
    {
        throw "Downloaded vcpkg.exe is not signed by Microsoft"
    }
    if (-not (Test-PinnedVersion $part))
    {
        throw "Downloaded vcpkg.exe is not version $tag"
    }
    Move-Item $part $tool -Force
}

if (Test-PinnedVersion $tool)
{
    Write-Host "vcpkg $tag is already bootstrapped"
}
else
{
    & "$vcpkgRoot/bootstrap-vcpkg.bat" -disableMetrics
    if ($LASTEXITCODE -ne 0)
    {
        Write-Host "Downloading vcpkg $tag through the GitHub API instead"
        Get-ThroughApi
    }
}
Set-Content -Path "$vcpkgRoot/vcpkg.disable-metrics" -Value ''
& $tool version --disable-metrics
