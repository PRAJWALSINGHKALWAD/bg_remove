@echo off
REM Activate virtual environment and run the background removal script
call "%~dp0venv\Scripts\activate"
python "%~dp0bg-remove.py" %*
