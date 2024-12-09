@echo off
SETLOCAL EnableDelayedExpansion

:: Set project directories
SET PROJECT_DIR=%~dp0
SET VENV_DIR=%PROJECT_DIR%.venv

:: Check if virtual environment exists
if not exist "%VENV_DIR%" (
    echo Virtual environment not found
    echo Please run setup.bat first to create the virtual environment.
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate"
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Verify Python environment
python -c "import cv2; import dlib; import numpy; import screeninfo" > nul 2>&1
if errorlevel 1 (
    echo Error: Required packages not found
    echo Please run setup.bat to install required packages
    pause
    exit /b 1
)

:: Run the program
echo Starting Face Recognition Program...
python "%PROJECT_DIR%FaceRecog.py"
if errorlevel 1 (
    echo Error: Program exited with an error
    pause
    exit /b 1
)

:: Deactivate virtual environment
call deactivate

ENDLOCAL
pause
