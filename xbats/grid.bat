@echo off
cd ..

echo ========================================
echo Building project...
echo ========================================
cmake --build build > nul

echo.
echo ========================================
echo Launching Broadphase Grid Test...
echo ========================================

cd build\bin\Debug
.\gridt.exe

echo.
pause
