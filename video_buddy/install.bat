@echo off
rem VIDEO BUDDY — Windows installer
cd /d "%~dp0"
where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python install.py %*
  exit /b %ERRORLEVEL%
)
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 install.py %*
  exit /b %ERRORLEVEL%
)
echo FAIL  Python 3.10+ not found. Install from https://www.python.org/downloads/ and tick Add Python to PATH.
exit /b 1
