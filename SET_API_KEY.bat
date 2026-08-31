@echo off
setlocal
cd /d "%~dp0"
title StressViz - assistant setup

echo.
echo  ===============================================
echo   StressViz - set up the Engineering Assistant
echo  ===============================================
echo.
echo  1. Open  https://console.groq.com/keys  in your browser
echo  2. Sign in (free), click "Create API Key", copy it
echo  3. Right-click here to paste it, then press Enter
echo.
set "GK="
set /p GK=Groq API key (starts with gsk_):

if "%GK%"=="" (
  echo.
  echo  Nothing entered - no changes made.
  echo.
  pause
  exit /b 1
)

echo.
echo  ===============================================
echo  Now the SEARCH key - strongly recommended.
echo  ===============================================
echo.
echo  Without it the assistant answers from memory only, with no citations
echo  and no access to FRCDesign.org, Chief Delphi, MatWeb or vendor docs.
echo.
echo  4. Open  https://tavily.com  and sign up (free tier is plenty)
echo  5. Copy the API key, right-click here to paste, press Enter
echo.
set "TK="
set /p TK=Tavily API key (starts with tvly-):

REM --- write .env (this file is git-ignored; keys never reach the browser) ---
> .env echo # StressViz secrets - keep this file private, never commit or share it.
>> .env echo GROQ_API_KEY=%GK%
>> .env echo GROQ_MODEL=openai/gpt-oss-120b
>> .env echo TAVILY_API_KEY=%TK%

echo.
if "%TK%"=="" (
  echo  NOTE: no search key entered - answers will have no sources.
)
echo.
echo  Saved to .env
echo.
echo  Now close the black StressViz server window if it is open,
echo  then run START_APP.bat again and refresh the page.
echo.
pause
