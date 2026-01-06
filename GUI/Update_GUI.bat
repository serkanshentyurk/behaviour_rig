@echo off
pushd "%~dp0.."

if not exist ".git" (
    echo ERROR: Not a git repository.
    pause
    exit /b 1
)

git pull
echo Successful

pause