@echo off
setlocal enabledelayedexpansion
REM ====== StressViz - calibrate pocketing against parts you have made ======
REM StressViz's pocketing is geometry with constants in it, not a trained model.
REM The constants were chosen by eye off parts in the 200-400 mm range, which is
REM why bigger plates come out with too many small pockets. This tool measures
REM plates you have actually machined and moves those constants toward them.
REM
REM Put reference parts in the reference_parts folder. Photos, screenshots or
REM STEP files. Name each one with its real size: bellypan_610mm.png
cd /d "%~dp0"
title StressViz pocket calibration

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo  Run START_APP.bat once first - it builds the environment this needs.
  echo.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"

REM --- dragged a part onto the icon? copy it into reference_parts ---
if not "%~1"=="" (
  if exist "%~1" (
    if not exist "reference_parts" mkdir "reference_parts"
    copy /y "%~1" "reference_parts\" >nul
    echo.
    echo   Copied "%~nx1" into reference_parts.
    echo.
    echo   If its name does not end in the part's real size - like _610mm -
    echo   rename it now, or StressViz has no idea what scale it is looking at.
    echo.
    pause
    exit /b 0
  )
  python tools\pocket_ref.py %*
  echo.
  pause
  exit /b 0
)

:menu
cls
echo.
echo   ===============================================
echo    StressViz - pocket calibration
echo   ===============================================
echo.
echo    1  What reference parts do I have?
echo    2  Compare my parts against what StressViz would cut
echo    3  Tune the constants to match my parts
echo    4  Show the calibration that is running now
echo    5  Go back to the shipped defaults
echo    6  Open the reference_parts folder
echo    0  Done
echo.
set "PICK="
set /p "PICK=   Choose: "

if "%PICK%"=="1" (
  python tools\pocket_ref.py list
  goto pause_menu
)
if "%PICK%"=="2" (
  echo.
  echo   Every measurement is divided by the part's own size before the two
  echo   are compared, so a gusset and a bellypan can sit in the same table.
  echo   Nothing is changed by this - it only measures and prints.
  echo.
  python tools\pocket_ref.py report
  goto pause_menu
)
if "%PICK%"=="3" (
  echo.
  echo   This searches the constants for the values that come closest to your
  echo   parts, and writes them to data\pocket_cal.json. It takes a few minutes
  echo   and runs the pocketing engine a few hundred times.
  echo.
  echo   Two honest warnings. Constants fitted to a handful of parts are a
  echo   guess with a decimal point on it, and they are a better guess only for
  echo   parts the size of the ones you gave it. And this changes what every
  echo   future pocketing run produces - option 5 puts it back.
  echo.
  set "OK="
  set /p "OK=   Show me what it would change, without writing it? (Y/n): "
  if /i "!OK!"=="n" (
    python tools\pocket_ref.py fit
  ) else (
    python tools\pocket_ref.py fit --dry-run
    echo.
    set "GO="
    set /p "GO=   Run it for real and write the file? (y/N): "
    if /i "!GO!"=="y" python tools\pocket_ref.py fit
  )
  goto pause_menu
)
if "%PICK%"=="4" (
  python tools\pocket_ref.py show
  goto pause_menu
)
if "%PICK%"=="5" (
  echo.
  echo   This deletes data\pocket_cal.json. The defaults live in the code, so
  echo   afterwards this install cuts exactly what a fresh one would.
  echo.
  set "OK="
  set /p "OK=   Go back to the defaults? (y/N): "
  if /i "!OK!"=="y" python tools\pocket_ref.py revert
  goto pause_menu
)
if "%PICK%"=="6" (
  if not exist "reference_parts" mkdir "reference_parts"
  start "" "reference_parts"
  goto menu
)
if "%PICK%"=="0" goto done
goto menu

:pause_menu
echo.
echo   ---
pause
goto menu

:done
echo.
echo   Calibration is read at the start of every pocketing run, so there is
echo   nothing to restart.
echo.
pause
