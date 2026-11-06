@echo off
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python -m pip install -r requirements-lock.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python -m pip install --no-deps -e .
if errorlevel 1 exit /b 1
.venv\Scripts\python -m pip check
echo Setup complete. Train with: .venv\Scripts\python train.py --mode full
