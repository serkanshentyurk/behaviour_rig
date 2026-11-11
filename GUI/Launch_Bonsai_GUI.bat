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

REM --- Work in GUI/, where this .bat lives (git finds the repository from here) ---
cd /d "%~dp0"

REM --- Sync to server: take origin/main exactly, discard local edits to
REM     TRACKED files. Ignored files (Rig_Params.csv, Subject_Params.csv) are
REM     left untouched, so this rig keeps its generated params. No "git clean":
REM     that would delete untracked files, which we never want on a rig.
REM     If the sync fails, say so and ask before starting with the code already
REM     on this PC: stale code is never used without someone choosing to.
set "SYNCED=1"
git fetch origin
if errorlevel 1 set "SYNCED=0"
if "%SYNCED%"=="1" git reset --hard origin/main
if errorlevel 1 set "SYNCED=0"

if "%SYNCED%"=="1" goto synced
echo.
echo ================================================================
echo  WARNING: could not update the code from GitHub.
echo  The code and mouse parameters on this PC are from:
git log -1 --format="    commit %%h, %%cd: %%s" --date=format:"%%Y-%%m-%%d %%H:%%M"
echo  Check the network, or see INSTRUCTIONS.md, Troubleshooting.
echo ================================================================
choice /C YN /M "Start the panel with this code anyway"
if errorlevel 2 goto end
goto launch_gui

:synced
echo Code up to date:
git log -1 --format="    commit %%h, %%cd: %%s" --date=format:"%%Y-%%m-%%d %%H:%%M"

:launch_gui
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
