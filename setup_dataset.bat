@echo off
cd /d "%~dp0"
echo ============================================
echo Face Recognition - Verify Dataset and Train
echo ============================================
echo.
echo Expected:
echo   dataset\ME\
echo   dataset\NOT_ME\
echo.

python scripts\import_dataset.py
if errorlevel 1 (
    echo.
    echo Add images to dataset\ME and dataset\NOT_ME then run again.
    pause
    exit /b 1
)

python run_pipeline.py
if errorlevel 1 pause & exit /b 1

echo.
echo ============================================
echo DONE. Start: python app\flask_app.py
echo ============================================
pause
