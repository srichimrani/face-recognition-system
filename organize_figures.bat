@echo off
cd /d "%~dp0"
echo Copying charts and organizing figures for report...
python scripts\organize_figures.py
echo.
echo If Flask screenshots are missing, save them as:
echo   docs\figures\incoming\flask_me.png
echo   docs\figures\incoming\flask_not_me.png
echo Then run this batch file again.
pause
