@echo off

REM --- Try activating Anaconda ---
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" gui_env
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" gui_env
) else (
    echo ERROR: Could not find Anaconda or Miniconda installation.
    pause
    exit /b
)


REM --- Move to the directory of this .bat file ---
cd /d "%~dp0"

REM --- Run the Python script that is in the same folder ---
python "Bonsai_GUI_Script_V2.py"