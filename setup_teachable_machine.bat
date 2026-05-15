@echo off
cd /d "%~dp0"
echo ============================================
echo Teachable Machine Setup
echo ============================================
echo.
echo 1. Train at https://teachablemachine.withgoogle.com/
echo 2. Export: TensorFlow -^> Keras -^> Download ZIP
echo 3. Drag ZIP onto this window OR enter path below
echo.
set /p ZIPPATH="ZIP file path: "
if "%ZIPPATH%"=="" (
    echo Running verify only...
    python scripts\setup_teachable_machine.py --verify
) else (
    python scripts\setup_teachable_machine.py "%ZIPPATH%"
)
echo.
echo Next: python scripts\compare_ml_tm.py
echo Then:  python app\flask_app.py
pause
