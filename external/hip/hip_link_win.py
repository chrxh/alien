#!/usr/bin/env python3
"""
Windows HIP device-link helper.

CMake's HIP link rule on Windows injects -fuse-ld=lld-link and uses
MSVC-style flags (/subsystem:console etc.) that conflict with or are not
understood by clang++ in --hip-link (device-link) mode. This script:
  - Strips -fuse-ld=* (conflicts with --hip-link)
  - Converts -Xlinker <flag> pairs and bare MSVC /flags to -Wl,<flag> form
  - Converts -lXXX.lib (GCC-style lib refs that lld-link ignores) to
    -Xlinker XXX.lib (so lld-link finds them via LIB env)
  - Wraps bare .lib names with -Xlinker for the GCC driver -> lld-link path

Usage (set as CMAKE_HIP_LINK_EXECUTABLE):
  python3 <this_script> <clang_exe> [args...]
"""
import subprocess
import sys


def transform(args):
    out = []
    i = 0
    while i < len(args):
        a = args[i]
        # CMake injects -fuse-ld=lld-link (Windows-Clang platform default)
        # into <LINK_FLAGS>; it conflicts with --hip-link. Drop it.
        if a.startswith('-fuse-ld='):
            i += 1
            continue
        # -Xlinker <flag> pairs from CMake: convert to -Wl,<flag> which is
        # handled correctly in --hip-link mode.
        if a == '-Xlinker' and i + 1 < len(args):
            out.append('-Wl,' + args[i + 1])
            i += 2
            continue
        # -lXXX.lib: GCC-driver lib syntax for Windows import libs. lld-link
        # ignores -lXXX.lib. Convert to -Xlinker XXX.lib so lld-link finds
        # the lib in the LIB env path.
        if a.startswith('-l') and a.endswith('.lib'):
            lib_name = a[2:]  # strip -l prefix
            out += ['-Xlinker', lib_name]
            i += 1
            continue
        # Bare MSVC-style flags (start with / and no file extension)
        base = a.split(':')[0]
        if a.startswith('/') and '.' not in base:
            out.append('-Wl,' + a)
        # Bare .lib names (no directory separators): wrap for GCC driver
        elif a.endswith('.lib') and '/' not in a and '\\' not in a:
            out += ['-Xlinker', a]
        else:
            out.append(a)
        i += 1
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: hip_link_win.py <clang_exe> [args...]")
    clang = sys.argv[1]
    transformed = transform(sys.argv[2:])
    sys.exit(subprocess.run([clang] + transformed).returncode)


if __name__ == '__main__':
    main()
