@echo off
setlocal enabledelayedexpansion
REM ====== StressViz — add documents to the assistant's library ======
REM Everything you add here is searched BEFORE the web, cited by name, and works
REM with no API key and no internet. The library is one file: data\kb.json.
REM
REM Three ways to use this:
REM   * Drag a PDF, .md, .txt or saved .html file onto this .bat  -> it's added
REM   * Double-click it                                          -> menu
REM   * From a terminal: ADD_TO_LIBRARY.bat url https://...       -> passed straight through
cd /d "%~dp0"
title StressViz library

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo  Run START_APP.bat once first - it builds the environment this needs.
  echo.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"

REM --- dragged a file onto the icon? %~1 is a real path, so ingest it ---
if not "%~1"=="" (
  if exist "%~1" (
    echo  Adding "%~1" to your library...
    python tools\kb_ingest.py file "%~1"
    echo.
    pause
    exit /b 0
  )
  REM Not a file: treat the whole command line as kb_ingest arguments.
  python tools\kb_ingest.py %*
  echo.
  pause
  exit /b 0
)

:menu
cls
echo.
echo   ===============================================
echo    StressViz - the assistant's document library
echo   ===============================================
echo.
python tools\kb_ingest.py stats
echo.
echo    1  Add a starter set of FRC design references (needs internet)
echo    2  Add all of FRCDesign.org - handbook, course, mechanisms (needs internet)
echo    3  Add a web page
echo    4  Add everything a resource page links to (FIRST tech resources)
echo    5  Add a file from this computer (PDF, notes, saved web page)
echo    6  Add a note you type yourself
echo    7  See what's in the library
echo    8  Test what the assistant would find for a question
echo    9  Check a PDF before adding it (does it actually have text?)
echo   10  Remove a document
echo    0  Done
echo.
set "PICK="
set /p "PICK=   Choose: "

if "%PICK%"=="1" (
  python tools\kb_ingest.py seed
  goto pause_menu
)
if "%PICK%"=="2" (
  echo.
  echo   This reads frcdesign.org properly: the design handbook, every section
  echo   of the learning course, the worked mechanism examples and the CAD
  echo   best practices - a few hundred pages. It takes several minutes.
  echo.
  echo   The course pages are tagged as it goes. A page that teaches motors or
  echo   ball trajectory is general engineering and is quoted as such; a page
  echo   that sets a design challenge has limits true only inside that
  echo   challenge, and gets tagged so the assistant says so instead of
  echo   repeating one exercise's numbers as a rule.
  echo.
  echo   Re-running this refreshes rather than duplicates, so it is safe to
  echo   run again next season.
  echo.
  set "OK="
  set /p "OK=   Go ahead? (y/N): "
  if /i "!OK!"=="y" python tools\kb_ingest.py frcdesign
  goto pause_menu
)
if "%PICK%"=="3" (
  set "U="
  set /p "U=   Paste the web address: "
  if not "!U!"=="" python tools\kb_ingest.py url "!U!"
  goto pause_menu
)
if "%PICK%"=="4" (
  echo.
  echo   A resource page is a list of documents, not a document. This adds the
  echo   things it links to - the PDFs and the sites - and not the list itself,
  echo   whose own text is link labels that match every question and answer none.
  echo.
  echo   Example: https://www.firstinspires.org/resources/library/frc/technical-resources
  set "U="
  set /p "U=   Paste the resource page address: "
  if "!U!"=="" goto pause_menu
  echo.
  echo   Dry run first - what would come in, and what would be left out and why.
  echo   Nothing is added yet.
  python tools\kb_ingest.py hub "!U!" --dry-run
  echo.
  set "OK="
  set /p "OK=   Add these to the library now? (y/N): "
  if /i "!OK!"=="y" python tools\kb_ingest.py hub "!U!"
  goto pause_menu
)
if "%PICK%"=="5" (
  echo   Tip: you can also just drag the file onto ADD_TO_LIBRARY.bat.
  set "F="
  set /p "F=   Full path to the file: "
  REM Quotes are stripped in case the path was pasted from "Copy as path",
  REM which wraps it in quotes that would then be doubled below.
  set "F=!F:"=!"
  if not "!F!"=="" python tools\kb_ingest.py file "!F!"
  goto pause_menu
)
if "%PICK%"=="6" (
  echo.
  echo   Tag this as your team's own convention, so the assistant says it is
  echo   YOUR practice rather than presenting it as a universal rule.
  set "T="
  set "B="
  set /p "T=   Short title: "
  set /p "B=   The note: "
  if not "!B!"=="" python tools\kb_ingest.py text "!B!" --title "!T!" --kind convention --source "team notes"
  goto pause_menu
)
if "%PICK%"=="7" (
  python tools\kb_ingest.py list
  goto pause_menu
)
if "%PICK%"=="8" (
  set "Q="
  set /p "Q=   Question: "
  if not "!Q!"=="" python tools\kb_ingest.py search "!Q!"
  goto pause_menu
)
if "%PICK%"=="9" (
  echo.
  echo   A scanned PDF has no text in it. It adds without complaint and holds
  echo   nothing. This shows you what would actually go in, and adds nothing.
  set "F="
  set /p "F=   Full path to the PDF: "
  set "F=!F:"=!"
  if not "!F!"=="" python tools\kb_ingest.py pdfcheck "!F!"
  goto pause_menu
)
if "%PICK%"=="10" (
  python tools\kb_ingest.py list
  set "R="
  set /p "R=   Paste the url/id to remove: "
  if not "!R!"=="" python tools\kb_ingest.py remove "!R!"
  goto pause_menu
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
echo   Restart StressViz (or just ask it something) to use the updated library.
echo.
pause
