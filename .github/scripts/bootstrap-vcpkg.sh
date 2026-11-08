#!/usr/bin/env bash
# Bootstraps the vcpkg tool pinned by the submodule. The release download from github.com fails
# now and then (HTTP 504) while the GitHub API still serves the same asset.
set -euo pipefail

vcpkgRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../external/vcpkg" && pwd)"
. "$vcpkgRoot/scripts/vcpkg-tool-metadata.txt"
tool="$vcpkgRoot/vcpkg"
trap 'rm -f "$tool.part"' EXIT

hasPinnedHash()
{
    [ -f "$1" ] && echo "$VCPKG_GLIBC_SHA  $1" | sha512sum --check --status
}

downloadThroughApi()
{
    local auth=()
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
    fi
    local assetUrl
    assetUrl=$(curl -fsSL --retry 5 "${auth[@]}" "https://api.github.com/repos/microsoft/vcpkg-tool/releases/tags/$VCPKG_TOOL_RELEASE_TAG" |
        jq -er '.assets[] | select(.name == "vcpkg-glibc") | .url')
    curl -fsSL --retry 5 "${auth[@]}" -H "Accept: application/octet-stream" -o "$tool.part" "$assetUrl"
    if ! hasPinnedHash "$tool.part"; then
        echo "Downloaded vcpkg does not match VCPKG_GLIBC_SHA" >&2
        exit 1
    fi
    chmod +x "$tool.part"
    mv "$tool.part" "$tool"
}

if hasPinnedHash "$tool"; then
    echo "vcpkg $VCPKG_TOOL_RELEASE_TAG is already bootstrapped"
elif ! "$vcpkgRoot/bootstrap-vcpkg.sh" -disableMetrics; then
    echo "Downloading vcpkg $VCPKG_TOOL_RELEASE_TAG through the GitHub API instead"
    downloadThroughApi
fi
touch "$vcpkgRoot/vcpkg.disable-metrics"
"$tool" version --disable-metrics
