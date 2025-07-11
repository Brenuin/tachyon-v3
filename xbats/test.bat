@echo off
cd ..
cmake --build build > nul

echo ============================
echo Launching GLFW Test Window...
echo ============================

cd build\bin\Debug
.\main_test.exe
pause
