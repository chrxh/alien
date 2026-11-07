@echo off
rem Build the Windows installer of ALIEN with Inno Setup 6.3 or newer
rem (https://jrsoftware.org/isdl.php).
rem
rem The installer packages the Release build of build-windows-ninja.bat, so build
rem ALIEN first. The AMD executable joins the installer if the HIP build exists:
rem
rem     build-windows-ninja.bat
rem     build-windows-ninja.bat HIP          (optional)
rem     installer\build-installer.bat
rem
rem The installer is written to installer\build\alien-setup.exe. The build tree is
rem chosen as in build-windows-ninja.bat: "ide" (build-ninja, build-ninja-hip) or
rem "agent" (build-agent, build-agent-hip). Without an argument the IDE tree is
rem used, except when CLAUDECODE is set. ALIEN_BUILD_TREE=ide or agent overrides
rem the default; an explicit argument wins over both.
rem
setlocal

for %%i in ("%~dp0..") do set "ROOT=%%~fi"

set "TREE=ide"
if "%CLAUDECODE%"=="1" set "TREE=agent"
if /i "%ALIEN_BUILD_TREE%"=="ide" set "TREE=ide"
if /i "%ALIEN_BUILD_TREE%"=="agent" set "TREE=agent"
if /i "%~1"=="ide" set "TREE=ide"
if /i "%~1"=="agent" set "TREE=agent"

if /i "%TREE%"=="agent" (
    set "NVIDIA_DIR=%ROOT%\build-agent\Release"
    set "AMD_DIR=%ROOT%\build-agent-hip\Release"
) else (
    set "NVIDIA_DIR=%ROOT%\build-ninja\Release"
    set "AMD_DIR=%ROOT%\build-ninja-hip\Release"
)

if not exist "%NVIDIA_DIR%\alien.exe" (
    echo [installer] "%NVIDIA_DIR%\alien.exe" was not found. Build ALIEN first.
    exit /b 1
)
if not exist "%AMD_DIR%\alien.exe" (
    echo [installer] No AMD build in "%AMD_DIR%", the installer covers NVIDIA GPUs only.
)

rem Inno Setup 6 is preferred because the script was tested with it.
set "ISCC="
for %%v in (6 7) do for %%d in ("%LOCALAPPDATA%\Programs" "%ProgramFiles(x86)%" "%ProgramFiles%") do (
    if not defined ISCC if exist "%%~d\Inno Setup %%v\ISCC.exe" set "ISCC=%%~d\Inno Setup %%v\ISCC.exe"
)
if not defined ISCC for /f "delims=" %%i in ('where ISCC.exe 2^>nul') do set "ISCC=%%i"
if not defined ISCC (
    echo [installer] Inno Setup was not found. Install it from https://jrsoftware.org/isdl.php
    exit /b 1
)

"%ISCC%" "/DNvidiaDir=%NVIDIA_DIR%" "/DAmdDir=%AMD_DIR%" "%~dp0alien.iss" || exit /b 1
echo [installer] Done: "%~dp0build\alien-setup.exe"
