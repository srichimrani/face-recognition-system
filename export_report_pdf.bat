@echo off
cd /d "%~dp0"
echo ============================================
echo Export Report to PDF
echo ============================================
echo.
echo OPTION A - Microsoft Word (recommended):
echo   1. Open docs\REPORT.md in Word (File - Open)
echo   2. Fix image paths if needed (docs\figures\)
echo   3. File - Save As - PDF
echo.
echo OPTION B - Pandoc (if installed):
where pandoc >nul 2>&1
if %errorlevel%==0 (
    pandoc docs\REPORT.md -o docs\REPORT.pdf --resource-path=docs
    echo Created docs\REPORT.pdf
) else (
    echo Pandoc not installed. Use Word or Google Docs.
)
echo.
echo Before export:
python scripts\validate_submission.py
pause
