@echo off
setlocal enabledelayedexpansion
REM ====== StressViz launcher — auto-selects a compatible Python (3.10-3.12) ======
REM The scientific stack (triangle, opencv, gmsh) has no wheels for Python 3.13/3.14
REM yet, so this picks 3.12/3.11/3.10 via the "py" launcher and rebuilds the venv
REM if it was made with an unsupported version.
cd /d "%~dp0"
title StressViz server

REM --- find a supported Python (prefer 3.12, then 3.11, 3.10) ---
set "PYEXE="
for %%V in (3.12 3.11 3.10) do (
  if not defined PYEXE (
    py -%%V --version >nul 2>nul && set "PYEXE=py -%%V"
  )
)
if not defined PYEXE (
  echo.
  echo  No supported Python found ^(need 3.10, 3.11, or 3.12^).
  echo  Python 3.13/3.14 are too new for the science packages.
  echo  Install Python 3.12 from https://www.python.org/downloads/windows/
  echo  then double-click this file again.
  echo.
  pause
  exit /b 1
)
echo  Using !PYEXE!

REM --- if an existing venv uses an unsupported Python, rebuild it ---
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys;raise SystemExit(0 if sys.version_info<(3,13) else 1)" 2>nul
  if errorlevel 1 (
    echo  Existing environment uses an unsupported Python — rebuilding...
    rmdir /s /q .venv
  )
)
if not exist ".venv\Scripts\python.exe" (
  echo  Creating virtual environment with !PYEXE!...
  !PYEXE! -m venv .venv
)
call ".venv\Scripts\activate.bat"

REM --- ensure packages are installed ---
python -c "import uvicorn, fastapi, skfem, triangle, cv2, gmsh" 2>nul
if errorlevel 1 (
  echo  Installing packages ^(a few minutes the first time^)...
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo  Package install FAILED. Send the error above to Claude.
    echo.
    pause
    exit /b 1
  )
)

REM --- knowledge-base extras (checked separately) ---
REM These are in requirements.txt, but the check above only looks at the six
REM packages the server itself needs -- so an environment built before the
REM knowledge base existed passes that check and never installs these. They are
REM only used by tools\kb_ingest.py, so failing here must not stop the app.
REM The pypdf version matters, not just its presence: layout-aware extraction
REM arrives in 4.0 and app\pdftext.py needs it to un-interleave two-column PDFs
REM and find running headers. An environment built when 3.x was current passes a
REM plain import check and then reads every manual flat.
python -c "import bs4, pypdf; raise SystemExit(0 if int(pypdf.__version__.split('.')[0])>=4 else 1)" 2>nul
if errorlevel 1 (
  echo  Installing knowledge-base extras ^(optional^)...
  pip install -U beautifulsoup4 "pypdf>=4.0" >nul 2>nul
)

REM --- optional readers for Google Slides and Google Sheets ---
REM Kept out of requirements.txt for the same reason as pypardiso: python-pptx
REM pulls in lxml, and a machine where that wheel will not build must still get
REM a working app. app\gdocs.py degrades rather than failing -- without
REM python-pptx a deck is read from its PDF export, which keeps the words and
REM the slide numbers and loses the speaker notes; without openpyxl a workbook
REM is read as CSV, which is the first tab only. Both say so when it happens.
python -c "import pptx, openpyxl" 2>nul
if errorlevel 1 (
  echo  Installing the Google Slides/Sheets readers ^(optional^)...
  pip install python-pptx openpyxl >nul 2>nul
)

REM --- optional fast solver for the 3D solid stress map ---
REM Kept out of requirements.txt on purpose: it pulls in Intel's MKL runtime,
REM and if that wheel is unavailable the app must still start. Without it a 3D
REM solve takes about 40 seconds instead of about 12.
python -c "import pypardiso" 2>nul
if errorlevel 1 (
  echo  Installing the fast 3D solver ^(optional - skipped if unavailable^)...
  pip install pypardiso >nul 2>nul
)

REM --- open the browser once the server is actually up ---
start "" powershell -WindowStyle Hidden -Command ^
  "for($i=0;$i -lt 90;$i++){try{Invoke-WebRequest 'http://localhost:8000/api/health' -UseBasicParsing -TimeoutSec 1 ^|Out-Null; Start-Process 'http://localhost:8000'; break}catch{Start-Sleep -Milliseconds 800}}"

echo.
echo  Starting StressViz... the browser opens automatically when it's ready.
echo  Keep THIS window open while you use the app. Close it to stop the server.
echo.
python -m uvicorn app.main:app --port 8000

echo.
echo  The server stopped. If that was unexpected, read any error above.
pause
