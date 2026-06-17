@echo off
title OuiChat Launcher
cd /d "%~dp0"

echo  Starting OuiChat server...
start "OuiChat Server" cmd /k "call "%USERPROFILE%\anaconda3\Scripts\activate.bat" french_teacher && cd /d "%~dp0" && uvicorn api:app --port 8000"

echo  Waiting for server to be ready...
timeout /t 6 /nobreak >nul

echo  Opening Firefox...
start "" firefox --new-window http://localhost:8000

exit
