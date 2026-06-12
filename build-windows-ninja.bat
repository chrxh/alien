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
rem     build-windows-ninja.bat            (Release, default)
rem     build-windows-ninja.bat Debug
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

set "BUILD_PRESET=ninja-release"
if /i "%~1"=="Debug" set "BUILD_PRESET=ninja-debug"

cmake --preset ninja || exit /b 1
cmake --build --preset %BUILD_PRESET% || exit /b 1

echo [build] Done. Executables are under build-ninja\Release (or build-ninja\Debug).