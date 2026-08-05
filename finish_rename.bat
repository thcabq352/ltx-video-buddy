@echo off
REM finish_rename.bat — run AFTER closing the Kimi CLI session (and anything
REM else running from "kimi ltx"). Renames the project dir to video_buddy,
REM updates git-side references, commits and pushes.
cd /d "%~dp0"

echo [1/4] Renaming folder...
ren "kimi ltx" video_buddy || (echo FAILED: folder still locked. Close all sessions/terminals using it and re-run. & pause & exit /b 1)

echo [2/4] Updating .gitignore and README links...
powershell -NoProfile -Command "(Get-Content .gitignore -Raw) -replace 'kimi ltx','video_buddy' | Set-Content .gitignore -NoNewline"
powershell -NoProfile -Command "(Get-Content README.md -Raw) -replace 'kimi%%20ltx','video_buddy' | Set-Content README.md -NoNewline"

echo [3/4] Committing...
git add -A || (echo git add failed & pause & exit /b 1)
git commit -m "Rename project dir: kimi ltx -> video_buddy" || echo (nothing to commit?)

echo [4/4] Pushing...
git push

echo.
echo Done. Reopen your session in: %CD%\video_buddy
echo Reminders:
echo  - ComfyUI extra_model_paths.yaml already points at video_buddy (do not
echo    start ComfyUI from the old path).
echo  - If pip acts up in .venv after the rename, use: .venv\Scripts\python.exe -m pip
echo  - Update the master-agent MCP path in %%USERPROFILE%%\.hermes\config.yaml
pause
