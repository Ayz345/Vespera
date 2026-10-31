@echo off
rem Starts the Vespera site locally and opens it in the browser.
rem Needs Python with the packages in requirements.txt (pip install -r requirements.txt).
cd /d "%~dp0"
if not "%1"=="" set PORT=%1
if "%PORT%"=="" set PORT=5000
start "" http://127.0.0.1:%PORT%/
python App.py
