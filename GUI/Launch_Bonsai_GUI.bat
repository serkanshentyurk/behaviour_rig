@echo off
setlocal

echo === STARTING SCRIPT ===

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

echo Conda activation attempted
echo Python location:
where python
python --version

REM --- Move to script directory ---
cd /d "%~dp0"
echo Current directory: %CD%

REM --- Run Python ---
echo Running Python script...
python "Bonsai_GUI.py"
echo Python exited with code %ERRORLEVEL%

:end
echo.
echo === PRESS ANY KEY TO CLOSE ===
pause >nul
