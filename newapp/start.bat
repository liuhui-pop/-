@echo off
set "ROOT=%~dp0"
set /p ENV_NAME=Enter conda env name (default: H, press Enter to skip): 
if "%ENV_NAME%"=="" set "ENV_NAME=H"

set "CONDA_BAT="
for /f "delims=" %%I in ('where conda.bat 2^>nul') do (
    if not defined CONDA_BAT set "CONDA_BAT=%%I"
)

if not defined CONDA_BAT (
    for %%D in ("%ProgramData%\Anaconda3" "%ProgramData%\Miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3") do (
        if not defined CONDA_BAT if exist "%%~D\condabin\conda.bat" set "CONDA_BAT=%%~D\condabin\conda.bat"
        if not defined CONDA_BAT if exist "%%~D\Scripts\conda.bat" set "CONDA_BAT=%%~D\Scripts\conda.bat"
    )
)

if not defined CONDA_BAT (
    echo Cannot find conda.bat.
    echo Please install Anaconda/Miniconda or add conda to PATH.
    pause
    exit /b 1
)

echo Checking dependencies...
call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
    echo Failed to activate conda environment: %ENV_NAME%
    pause
    exit /b 1
)
pip install -r "%ROOT%backend\requirements.txt" -q

echo Checking frontend dependencies...
if not exist "%ROOT%node_modules" (
    echo Installing frontend packages...
    cd /d "%ROOT%"
    npm install
)

echo Starting backend...
start "Backend" /D "%ROOT%backend" cmd /k call "%CONDA_BAT%" activate "%ENV_NAME%" ^&^& uvicorn api:app --port 8000
timeout /t 5 /nobreak >nul

echo Starting frontend...
if exist "%ROOT%.next" rmdir /s /q "%ROOT%.next"
start "Frontend" /D "%ROOT%" cmd /k npm run dev
timeout /t 8 /nobreak >nul

start http://localhost:3000
