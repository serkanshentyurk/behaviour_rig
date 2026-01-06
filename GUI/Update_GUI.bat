@echo off
setlocal

REM --- Path to your master Rig_Params.csv ---
set MASTER_FILE=%USERPROFILE%\Documents\Rig_Params.csv

REM --- Path to the file inside the repo ---
set TARGET_FILE=%~dp0\Params\Rig_Params.csv

REM --- Try activating Conda ---
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
    echo Found Miniconda
    call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate gui_env
) else if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    echo Found Anaconda
    call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate gui_env
) else (
    echo ERROR: Conda not found
    goto end
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

echo Done!

:end
echo.
echo === PRESS ANY KEY TO CLOSE ===
pause >nul

