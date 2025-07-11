@echo off
cd ..

cmake --build build > nul

echo ========================================
echo Launching Box Rotation Test...
echo ========================================

cd build\bin\Debug
.\box_rotation_test.exe
pause