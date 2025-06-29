@echo off
REM Ultra-Quality Background Removal System (Maximum Quality)
echo Starting ULTRA-QUALITY processing...
echo Focus: Maximum Quality (processing time not important)
call "%~dp0venv\Scripts\activate"
python "%~dp0ultra_quality.py" %*
