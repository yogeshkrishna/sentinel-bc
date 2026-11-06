@echo off
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python -m pip install -r requirements-lock.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python -m pip install --no-deps -e .
if errorlevel 1 exit /b 1
.venv\Scripts\python -m pip check
echo Setup complete. Run run.bat and open http://127.0.0.1:7860.
