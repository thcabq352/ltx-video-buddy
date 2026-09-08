@echo off
rem VIDEO BUDDY studio dashboard - http://127.0.0.1:8189
cd /d "%~dp0"
start "" /min cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8189"
".venv\Scripts\python.exe" -m master_agent ui --host 127.0.0.1 --port 8189
pause
