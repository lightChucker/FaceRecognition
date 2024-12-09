@echo off
SETLOCAL EnableDelayedExpansion

:: Set project directories
SET PROJECT_DIR=%~dp0
SET VENV_DIR=%PROJECT_DIR%.venv
SET MODELS_DIR=%PROJECT_DIR%models

echo Setting up Face Recognition Project...
echo Project directory: %PROJECT_DIR%

:: Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or later and try again
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate"
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install required packages
echo Installing required packages...
pip install numpy
pip install opencv-python
pip install dlib
pip install screeninfo
pip install cmake

:: Create models directory if it doesn't exist
if not exist "%MODELS_DIR%" (
    echo Creating models directory...
    mkdir "%MODELS_DIR%"
)

:: Download model files if they don't exist
if not exist "%MODELS_DIR%\shape_predictor_68_face_landmarks.dat" (
    echo Downloading shape predictor model...
    powershell -Command "& {Invoke-WebRequest -Uri 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2' -OutFile '%MODELS_DIR%\shape_predictor_68_face_landmarks.dat.bz2'}"
    echo Extracting shape predictor model...
    python -c "import bz2; open('%MODELS_DIR%\shape_predictor_68_face_landmarks.dat', 'wb').write(bz2.open('%MODELS_DIR%\shape_predictor_68_face_landmarks.dat.bz2').read())"
    del "%MODELS_DIR%\shape_predictor_68_face_landmarks.dat.bz2"
)

if not exist "%MODELS_DIR%\dlib_face_recognition_resnet_model_v1.dat" (
    echo Downloading face recognition model...
    powershell -Command "& {Invoke-WebRequest -Uri 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2' -OutFile '%MODELS_DIR%\dlib_face_recognition_resnet_model_v1.dat.bz2'}"
    echo Extracting face recognition model...
    python -c "import bz2; open('%MODELS_DIR%\dlib_face_recognition_resnet_model_v1.dat', 'wb').write(bz2.open('%MODELS_DIR%\dlib_face_recognition_resnet_model_v1.dat.bz2').read())"
    del "%MODELS_DIR%\dlib_face_recognition_resnet_model_v1.dat.bz2"
)

:: Create requirements.txt if it doesn't exist
if not exist "%PROJECT_DIR%requirements.txt" (
    echo Creating requirements.txt...
    echo numpy==1.24.3 > requirements.txt
    echo opencv-python==4.8.1.78 >> requirements.txt
    echo dlib==19.24.2 >> requirements.txt
    echo screeninfo==0.8.1 >> requirements.txt
    echo cmake==3.27.7 >> requirements.txt
)

:: Check if installation was successful
python -c "import cv2; import dlib; import numpy; import screeninfo" > nul 2>&1
if errorlevel 1 (
    echo Error: Package installation failed
    pause
    exit /b 1
) else (
    echo All packages installed successfully!
)

:: Create the improved run.bat
echo Creating run script...
(
    echo @echo off
    echo SETLOCAL EnableDelayedExpansion
    echo.
    echo :: Set project directories
    echo SET PROJECT_DIR=%%~dp0
    echo SET VENV_DIR=%%PROJECT_DIR%%.venv
    echo.
    echo :: Check if virtual environment exists
    echo if not exist "%%VENV_DIR%%" ^(
    echo     echo Virtual environment not found!
    echo     echo Please run setup.bat first to create the virtual environment.
    echo     pause
    echo     exit /b 1
    echo ^)
    echo.
    echo :: Activate virtual environment
    echo echo Activating virtual environment...
    echo call "%%VENV_DIR%%\Scripts\activate"
    echo if errorlevel 1 ^(
    echo     echo Error: Failed to activate virtual environment
    echo     pause
    echo     exit /b 1
    echo ^)
    echo.
    echo :: Verify Python environment
    echo python -c "import cv2; import dlib; import numpy; import screeninfo" ^> nul 2^>^&1
    echo if errorlevel 1 ^(
    echo     echo Error: Required packages not found
    echo     echo Please run setup.bat to install required packages
    echo     pause
    echo     exit /b 1
    echo ^)
    echo.
    echo :: Run the program
    echo echo Starting Face Recognition Program...
    echo python "%%PROJECT_DIR%%FaceRecog.py"
    echo if errorlevel 1 ^(
    echo     echo Error: Program exited with an error
    echo     pause
    echo     exit /b 1
    echo ^)
    echo.
    echo :: Deactivate virtual environment
    echo call deactivate
    echo.
    echo ENDLOCAL
    echo pause
) > "%PROJECT_DIR%run.bat"

echo.
echo Setup complete! You can now run the program using run.bat
echo.
pause

ENDLOCAL
