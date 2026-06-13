#!/usr/bin/env bash
# Package ALIEN into a distributable tar.gz archive
# Usage: package-release.sh <version-tag> <build-dir>
# Example: package-release.sh v5.0.0 build

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <version-tag> <build-dir>"
    exit 1
fi

TAG="$1"
BUILD_DIR="$2"
PKG_DIR="${PWD}/alien-${TAG}-linux-x64"

echo "Creating package: ${PKG_DIR}"

# Create directory structure
mkdir -p "${PKG_DIR}/lib"

# Copy binaries
echo "Copying binaries..."
cp "${BUILD_DIR}/alien" "${PKG_DIR}/"
cp "${BUILD_DIR}/cli" "${PKG_DIR}/"

# Copy resources and docs
echo "Copying resources and docs..."
cp -r resources "${PKG_DIR}/"
cp imgui.ini "${PKG_DIR}/"
cp LICENSE "${PKG_DIR}/"
cp README.md "${PKG_DIR}/"
cp RELEASE-NOTES.md "${PKG_DIR}/"

# Bundle shared libraries
echo "Bundling shared libraries..."
bundle_libs() {
    local exe="$1"
    ldd "${BUILD_DIR}/${exe}" | grep '=>' | awk '{print $3}' | while read -r lib; do
        if [ -n "$lib" ] && [ -f "$lib" ]; then
            case "$lib" in
                */libc.so*|*/libdl.so*|*/libm.so*|*/libpthread.so*|*/librt.so*) ;;
                */libstdc++.so*|*/libgcc_s.so*|*/libz.so*) ;;
                */libGL.so*|*/libGLU.so*|*/libGLX.so*|*/libOpenGL.so*) ;;
                */libX11.so*|*/libXext.so*|*/libXfixes.so*|*/libXxf86vm.so*) ;;
                */libXi.so*|*/libXrandr.so*|*/libXrender.so*|*/libXcursor.so*|*/libXinerama.so*) ;;
                */libxcb.so*|*/libXau.so*|*/libXdmcp.so*) ;;
                */libffi.so*|*/libcuda.so*|*/libnvidia*) ;;
                *)
                    cp -n "$lib" "${PKG_DIR}/lib/" 2>/dev/null || true
                    echo "  bundled: $(basename "$lib")"
                    ;;
            esac
        fi
    done
}

bundle_libs alien
bundle_libs cli

# Also bundle vcpkg-installed shared libraries
if [ -d "${BUILD_DIR}/vcpkg_installed/x64-linux/lib" ]; then
    find "${BUILD_DIR}/vcpkg_installed/x64-linux/lib" -name "*.so*" -type f 2>/dev/null | while read -r lib; do
        cp -n "$lib" "${PKG_DIR}/lib/" 2>/dev/null || true
        echo "  bundled (vcpkg): $(basename "$lib")"
    done
fi

# Clean up empty lib directory
rmdir "${PKG_DIR}/lib" 2>/dev/null || true

# Create launch wrapper script if lib directory is non-empty
if [ -d "${PKG_DIR}/lib" ]; then
    echo "Creating launch wrapper script..."
    cat > "${PKG_DIR}/run.sh" << 'RUNSCRIPT'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"
exec "$SCRIPT_DIR/alien" "$@"
RUNSCRIPT
    chmod +x "${PKG_DIR}/run.sh"
fi

# Create tarball
echo "Creating tarball..."
PKG_FILE="alien-${TAG}-linux-x64.tar.gz"
tar czf "${PKG_FILE}" "$(basename "${PKG_DIR}")"

# Create checksums
echo "Creating checksums..."
cd "${PKG_DIR}"
find . -type f -exec sha256sum {} \; > "${OLDPWD}/SHA256SUMS"
cd "${OLDPWD}"

echo "Package created: ${PKG_FILE}"
echo "Package size: $(du -h "${PKG_FILE}" | cut -f1)"
