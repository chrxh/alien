@echo off
setlocal

rem Quellverzeichnis angeben
set source_dir=d:\dev\alien

rem Zielverzeichnis angeben
set destination_dir=d:\Alle wichtigen Dateien\Artificial Life\installer

del /q /s "%destination_dir%\bin\*.*"
rmdir /s /q "%destination_dir%\bin"

del /q /s "%destination_dir%\source\*.*"
rmdir /s /q "%destination_dir%\source"

del /q /s "%destination_dir%\external\*.*"
rmdir /s /q "%destination_dir%\external"

del /q /s "%destination_dir%\scripts\*.*"
rmdir /s /q "%destination_dir%\scripts"

mkdir "%destination_dir%\bin"
mkdir "%destination_dir%\bin\resources"

copy "%source_dir%\build\Release\alien.exe" "%destination_dir%\bin"
copy "%source_dir%\build\Release\brotlicommon.dll" "%destination_dir%\bin"
copy "%source_dir%\build\Release\brotlidec.dll" "%destination_dir%\bin"
copy "%source_dir%\build\Release\bz2.dll" "%destination_dir%\bin"
copy "%source_dir%\build\Release\cli.exe" "%destination_dir%\bin"
copy "%source_dir%\build\Release\EngineTests.exe" "%destination_dir%\bin"
copy "%source_dir%\build\Release\freetype.dll" "%destination_dir%\bin"
copy "%source_dir%\build\Release\glfw3.dll" "%destination_dir%\bin"
copy "%source_dir%\build\Release\gtest.dll" "%destination_dir%\bin"
copy "%source_dir%\build\Release\gtest_main.dll" "%destination_dir%\bin"
copy "%source_dir%\imgui.ini" "%destination_dir%\bin"
copy "%source_dir%\build\Release\libcrypto-3-x64.dll" "%destination_dir%\bin"
copy "%source_dir%\build\Release\libpng16.dll" "%destination_dir%\bin"
copy "%source_dir%\build\Release\libssl-3-x64.dll" "%destination_dir%\bin"
copy "%source_dir%\LICENSE" "%destination_dir%\bin"
copy "%source_dir%\build\Release\NetworkTests.exe" "%destination_dir%\bin"
copy "%source_dir%\RELEASE-NOTES.md" "%destination_dir%\bin"
copy "%source_dir%\README.md" "%destination_dir%\bin"
copy "%source_dir%\build\Release\vcredist_x64.exe" "%destination_dir%\bin"
copy "%source_dir%\build\Release\zlib1.dll" "%destination_dir%\bin"

copy "%source_dir%\resources\alien.ico" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\autosave.sim" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\autosave.settings.json" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\autosave.statistics.csv" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\ca-bundle.crt" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\editor off.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\editor on.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji1.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji2.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji3.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji4.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji5.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji6.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji7.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji8.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji9.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji10.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji11.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji12.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji13.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji14.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji15.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji16.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji17.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji18.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji19.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji20.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji21.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji22.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji23.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji24.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji25.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji26.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji27.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji28.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji29.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji30.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji31.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji32.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji33.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji34.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji35.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji36.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji37.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji38.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji39.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji40.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji41.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji42.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji43.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji44.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji45.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji46.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji47.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji48.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\emoji49.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\logo.png" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\shader.fs" "%destination_dir%\bin\resources"
copy "%source_dir%\resources\shader.vs" "%destination_dir%\bin\resources"

xcopy "%source_dir%\source" "%destination_dir%\source\" /s /e /y

mkdir "%destination_dir%\external"
xcopy "%source_dir%\external\cpp-httplib" "%destination_dir%\external\cpp-httplib\" /s /e /y
xcopy "%source_dir%\external\cuda" "%destination_dir%\external\cuda\" /s /e /y
xcopy "%source_dir%\external\Fonts" "%destination_dir%\external\Fonts\" /s /e /y
xcopy "%source_dir%\external\ImFileDialog" "%destination_dir%\external\ImFileDialog\" /s /e /y
xcopy "%source_dir%\external\WinReg" "%destination_dir%\external\WinReg\" /s /e /y

mkdir "%destination_dir%\scripts"
xcopy "%source_dir%\scripts" "%destination_dir%\scripts\" /s /e /y

copy "%source_dir%\CMakeLists.txt" "%destination_dir%\CMakeLists.txt"
copy "%source_dir%\LICENSE" "%destination_dir%\license.txt"
copy "%source_dir%\vcpkg.json" "%destination_dir%\vcpkg.json"

echo Fertig.
