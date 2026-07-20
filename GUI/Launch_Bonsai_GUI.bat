@echo off
setlocal

REM --- Activate the conda env ---
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

REM --- Move to the repo (this .bat lives in GUI/, so go up one) ---
cd /d "%~dp0"

REM --- Sync to server: take origin/main exactly, discard local edits to
REM     TRACKED files. Ignored files (Rig_Params.csv, Subject_Params.csv) are
REM     left untouched, so this rig keeps its generated params. No "git clean":
REM     that would delete untracked files, which we never want on a rig.
git fetch origin
git reset --hard origin/main

echo === STARTING GUI ===
where python
python --version

cd /d "%~dp0"
echo Current directory: %CD%
python "Bonsai_GUI.py"
echo Python exited with code %ERRORLEVEL%

:end
echo.
echo === PRESS ANY KEY TO CLOSE ===
pause >nul
