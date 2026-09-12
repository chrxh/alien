@echo off
rem Build ALIEN on Windows using the fast "Ninja Multi-Config" preset.
rem
rem This compiles all CUDA translation units in parallel and is considerably
rem faster than the default Visual Studio generator (which builds the CUDA
rem libraries one project after another).
rem
rem The script sets up the MSVC build environment automatically, so it can be run
rem from any prompt (PowerShell or cmd). An identical copy lives in the repository
rem root and in scripts\; either can be run directly:
rem
rem     build-windows-ninja.bat            (Release, default, all cores)
rem     build-windows-ninja.bat Debug
rem
rem An optional numeric argument caps the number of parallel build jobs. Without
rem it Ninja uses all available cores. Automated callers (e.g. a Claude agent)
rem should pass half the core count to keep the machine responsive:
rem
rem     build-windows-ninja.bat 16         (Release, 16 jobs)
rem     build-windows-ninja.bat Debug 16   (Debug, 16 jobs)
rem
rem The argument "HIP" builds the GPU engine for AMD GPUs via ROCm/HIP instead of
rem CUDA. It requires a HIP SDK installation (HIP_PATH) and writes to a separate
rem build tree, so a CUDA build in build-ninja is left untouched:
rem
rem     build-windows-ninja.bat HIP        (Release for AMD, into build-ninja-hip)
rem
rem There are two separate build trees. An IDE that opens this folder configures
rem the "ninja" preset into build-ninja and caches its model of that tree (Visual
rem Studio does so under .vs). A build from the outside regenerates that tree and
rem invalidates the cache, after which the IDE fails to build until .vs is deleted
rem and it is restarted. Automated builds therefore go to build-agent instead:
rem
rem     build-windows-ninja.bat ide        (into build-ninja, the IDE tree)
rem     build-windows-ninja.bat agent      (into build-agent)
rem
rem Without an argument the IDE tree is used, except when CLAUDECODE is set -- a
rem Claude Code agent builds into build-agent automatically. ALIEN_BUILD_TREE=ide
rem or agent overrides the default; an explicit argument wins over both.
rem
setlocal enabledelayedexpansion

rem Change to the repository root (this script may sit in the root or in scripts\).
cd /d "%~dp0"
if not exist "CMakePresets.json" cd ..

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [build] Could not find vswhere.exe. Is Visual Studio installed?
    exit /b 1
)

set "VSINSTALL="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
    echo [build] Could not find a Visual Studio installation with the C++ toolset.
    exit /b 1
)

call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1

rem Put the Ninja that ships with the "C++ CMake tools" VS component on PATH.
set "VSNINJA=%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if exist "%VSNINJA%\ninja.exe" set "PATH=%VSNINJA%;%PATH%"
where ninja >nul 2>nul || (
    echo [build] Ninja was not found. Install the "C++ CMake tools for Windows"
    echo [build] Visual Studio component, or put ninja.exe on PATH.
    exit /b 1
)

rem Prefer the CMake that ships with Visual Studio: it matches the installed MSVC
rem toolset. An older CMake on PATH can fail the Ninja generate step with
rem "No known features for CXX compiler MSVC" when the MSVC toolset is newer.
set "VSCMAKE=%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
if exist "%VSCMAKE%\cmake.exe" set "PATH=%VSCMAKE%;%PATH%"

rem Parse arguments in any order: "Debug"/"Release" selects the config, "HIP"
rem selects the AMD GPU backend, "ide"/"agent" selects the build tree, a numeric
rem argument caps the parallel build jobs (default: all cores).
set "CONFIG=Release"
set "USE_HIP="
set "JOBS="
set "TREE_ARG="
:parseargs
if "%~1"=="" goto :doneargs
if /i "%~1"=="Debug" set "CONFIG=Debug"
if /i "%~1"=="Release" set "CONFIG=Release"
if /i "%~1"=="HIP" set "USE_HIP=1"
if /i "%~1"=="ide" set "TREE_ARG=ide"
if /i "%~1"=="agent" set "TREE_ARG=agent"
echo %~1| findstr /r "^[1-9][0-9]*$" >nul && set "JOBS=%~1"
shift
goto :parseargs
:doneargs

rem Select the build tree: an explicit argument beats ALIEN_BUILD_TREE, which
rem beats the automatic choice (agent under Claude Code, otherwise the IDE tree).
set "TREE=ide"
if "%CLAUDECODE%"=="1" set "TREE=agent"
if /i "%ALIEN_BUILD_TREE%"=="ide" set "TREE=ide"
if /i "%ALIEN_BUILD_TREE%"=="agent" set "TREE=agent"
if defined TREE_ARG set "TREE=%TREE_ARG%"

if defined USE_HIP goto :hipbuild

if /i "%TREE%"=="agent" (
    set "CONFIGURE_PRESET=ninja-agent"
    set "PRESET_PREFIX=ninja-agent"
    set "OUTDIR=build-agent"
) else (
    set "CONFIGURE_PRESET=ninja"
    set "PRESET_PREFIX=ninja"
    set "OUTDIR=build-ninja"
)
if /i "%CONFIG%"=="Debug" (set "BUILD_PRESET=!PRESET_PREFIX!-debug") else (set "BUILD_PRESET=!PRESET_PREFIX!-release")
echo [build] Building into !OUTDIR! ^(%TREE% tree^).
cmake --preset !CONFIGURE_PRESET! || exit /b 1
if defined JOBS (
    echo [build] Limiting build to %JOBS% parallel jobs.
    cmake --build --preset %BUILD_PRESET% -j %JOBS% || exit /b 1
) else (
    cmake --build --preset %BUILD_PRESET% || exit /b 1
)

echo [build] Done. Executables are under !OUTDIR!\Release ^(or !OUTDIR!\Debug^).
exit /b 0

rem AMD GPUs: ALIEN_HIP_ARCH lists the target architectures. They all go into one
rem fat binary, so a single alien.exe runs on every listed GPU; the generic targets
rem below each cover a whole family (RDNA2, RDNA3, RDNA4) with one code object.
rem Set ALIEN_HIP_ARCH in the environment to build for a different set; configure
rem the ninja-hip preset directly to auto-detect the GPUs of the host instead.
:hipbuild
if not defined ALIEN_HIP_ARCH set "ALIEN_HIP_ARCH=gfx10-3-generic;gfx11-generic;gfx12-generic"
if /i "%TREE%"=="agent" (
    set "CONFIGURE_PRESET=ninja-hip-agent"
    set "PRESET_PREFIX=ninja-hip-agent"
    set "OUTDIR=build-agent-hip"
) else (
    set "CONFIGURE_PRESET=ninja-hip"
    set "PRESET_PREFIX=ninja-hip"
    set "OUTDIR=build-ninja-hip"
)
if /i "%CONFIG%"=="Debug" (set "BUILD_PRESET=!PRESET_PREFIX!-debug") else (set "BUILD_PRESET=!PRESET_PREFIX!-release")

rem find_package(hip) does not see the HIP SDK through the vcpkg toolchain unless
rem its install root is on CMAKE_PREFIX_PATH. HIP_PATH points into "Program Files"
rem and ends in a backslash, so the argument has to be quoted as a whole and the
rem trailing backslash dropped -- it would otherwise escape the closing quote.
set "HIP_ROOT=%HIP_PATH%"
if not defined HIP_ROOT (
    echo [build] HIP_PATH is not set. Install the AMD HIP SDK for Windows.
    exit /b 1
)
if "%HIP_ROOT:~-1%"=="\" set "HIP_ROOT=%HIP_ROOT:~0,-1%"
set "HIP_ROOT=%HIP_ROOT:\=/%"

rem CMake compares the compiler id across C, CXX and HIP and refuses a mixture
rem (Windows-Clang.cmake, __verify_same_language_values), so the whole build has to
rem use the clang-cl of the HIP SDK -- pointing only the HIP language at it fails.
set "HIP_CLANG_CL=%HIP_ROOT%/bin/clang-cl.exe"

rem The CMake bundled with Visual Studio (3.31) derives MSVC_VERSION only from the
rem C / CXX / CUDA simulate version and never from CMAKE_HIP_SIMULATE_VERSION, so
rem its HIP-only ABI check aborts with "MSVC compiler version not detected
rem properly". A standalone CMake 4.x covers the HIP case; prefer it here.
set "CMAKE_EXE=cmake"
if exist "%ProgramFiles%\CMake\bin\cmake.exe" set "CMAKE_EXE=%ProgramFiles%\CMake\bin\cmake.exe"
rem Route the version through a file: a quoted program path inside for /f gets
rem mangled by the cmd /c that runs the nested command.
set "CMAKE_VER="
"%CMAKE_EXE%" --version > "%TEMP%\alien-cmake-version.txt" 2>nul
for /f "tokens=3" %%v in ('findstr /r /c:"^cmake version" "%TEMP%\alien-cmake-version.txt"') do set "CMAKE_VER=%%v"
del "%TEMP%\alien-cmake-version.txt" >nul 2>nul
for /f "tokens=1 delims=." %%v in ("%CMAKE_VER%") do set "CMAKE_MAJOR=%%v"
if not defined CMAKE_MAJOR set "CMAKE_MAJOR=0"
if %CMAKE_MAJOR% LSS 4 (
    echo [build] The AMD build needs CMake 4.0 or newer, found "%CMAKE_VER%".
    echo [build] Install it from https://cmake.org/download/ ^(the CMake shipped with
    echo [build] Visual Studio does not detect MSVC_VERSION for the HIP language^).
    exit /b 1
)

echo [build] AMD/ROCm build for architectures: %ALIEN_HIP_ARCH% ^(CMake %CMAKE_VER%, into !OUTDIR!^)
"%CMAKE_EXE%" --preset !CONFIGURE_PRESET! -DCMAKE_HIP_ARCHITECTURES="%ALIEN_HIP_ARCH%" "-DCMAKE_PREFIX_PATH=%HIP_ROOT%" "-DCMAKE_C_COMPILER=%HIP_CLANG_CL%" "-DCMAKE_CXX_COMPILER=%HIP_CLANG_CL%" "-DCMAKE_HIP_COMPILER=%HIP_CLANG_CL%" || exit /b 1
if defined JOBS (
    echo [build] Limiting build to %JOBS% parallel jobs.
    "%CMAKE_EXE%" --build --preset %BUILD_PRESET% -j %JOBS% || exit /b 1
) else (
    "%CMAKE_EXE%" --build --preset %BUILD_PRESET% || exit /b 1
)

echo [build] Done. Executables are under !OUTDIR!\%CONFIG%.