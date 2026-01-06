@echo off
setlocal

REM --- Path to your master Rig_Params.csv ---
set MASTER_FILE=%USERPROFILE%\Documents\Rig_Params.csv

REM --- Path to the file inside the repo ---
set TARGET_FILE=%~dp0\Params\Rig_Params.csv

REM --- Activate Conda environment ---
call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate gui_env
if errorlevel 1 (
    echo ERROR: Conda activation failed.
    pause
    exit /b
)

REM --- Move to the directory of this .bat file (repo root) ---
cd /d "%~dp0"

REM --- Pull the latest changes ---
git pull

REM --- Restore the protected file from master location ---
if exist "%MASTER_FILE%" (
    copy /Y "%MASTER_FILE%" "%TARGET_FILE%"
    echo Protected file restored from %MASTER_FILE%.
) else (
    echo WARNING: Master file not found at %MASTER_FILE%.
)

REM --- Run your Python GUI ---
python "Bonsai_GUI_Script_V2.py"

echo Done!
pause
