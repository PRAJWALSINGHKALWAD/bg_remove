@echo off
REM Remove.bg Quality Background Removal System
call "%~dp0venv\Scripts\activate"
python "%~dp0removebg_quality.py" %*
