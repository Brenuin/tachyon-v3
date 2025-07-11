@echo off
cd ..

echo ========================================
echo Building project...
echo ========================================
cmake --build build > nul

echo.
echo ========================================
echo Launching Broadphase contact_test...
echo ========================================

cd build\bin\Debug
.\contact_test.exe

echo.
pause
