@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "MODEL_NAME=uvr-mdx-hq-3.onnx"
set "TEMP_MODEL=0"
set "MODEL_PATH="

echo ========================================================
echo Building NOT INTEGRATED version (AI Models Manager)...
echo ========================================================
cargo build --release
if errorlevel 1 goto :error
copy /Y target\release\rustvocalremover.exe rustvocalremover_notintegrated.exe >nul

echo.
if exist "%MODEL_NAME%" goto :build_integrated

echo Looking for %MODEL_NAME% in AI Models Manager...
for /f "tokens=2,*" %%A in ('reg query "HKLM\SOFTWARE\Gallery Inc\AI-ModelsManager\models" /v "%MODEL_NAME%" 2^>nul ^| find /I "%MODEL_NAME%"') do set "MODEL_PATH=%%B"

if not defined MODEL_PATH goto :skip_integrated
if not exist "%MODEL_PATH%" goto :skip_integrated

copy /Y "%MODEL_PATH%" "%MODEL_NAME%" >nul
if errorlevel 1 goto :skip_integrated
set "TEMP_MODEL=1"
echo Model copied temporarily from AI Models Manager.

:build_integrated
echo ========================================================
echo Building INTEGRATED version (Embedded Model)...
echo ========================================================
cargo build --release --features integrated
if errorlevel 1 goto :error
copy /Y target\release\rustvocalremover.exe rustvocalremover_integrated.exe >nul
if "%TEMP_MODEL%"=="1" del /Q "%MODEL_NAME%" >nul 2>&1
goto :done

:skip_integrated
echo ========================================================
echo INTEGRATED build skipped.
echo Could not find %MODEL_NAME% locally or in AI Models Manager.
echo ========================================================
if "%TEMP_MODEL%"=="1" del /Q "%MODEL_NAME%" >nul 2>&1
goto :done

:done
echo.
echo ========================================================
echo Build complete.
echo   rustvocalremover_notintegrated.exe = uses AI Models Manager
if exist "rustvocalremover_integrated.exe" echo   rustvocalremover_integrated.exe    = model embedded

echo.
echo IMPORTANT after upgrading from an older build:
echo Reinstall the Explorer Actions menu once, either from the GUI button
echo or with the executable you want Explorer to use:
echo   rustvocalremover_notintegrated.exe --add
echo   rustvocalremover_integrated.exe --add

echo ========================================================
pause
exit /b 0

:error
if "%TEMP_MODEL%"=="1" del /Q "%MODEL_NAME%" >nul 2>&1
echo.
echo Build failed.
pause
exit /b 1
