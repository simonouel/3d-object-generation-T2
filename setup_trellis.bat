@echo off

REM Setup trellis2 submodule (MIT licensed, no patches required)

if exist "trellis\.git" (
    echo TRELLIS 2 submodule already set up.
    exit /b 0
)

if exist "trellis" (
    echo Removing incomplete trellis directory...
    rmdir /s /q trellis >nul 2>&1
)

git submodule init trellis >nul 2>&1
git submodule update --init trellis >nul 2>&1

echo TRELLIS 2 submodule setup complete.
exit /b 0
