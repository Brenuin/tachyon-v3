@echo off
cd ..
cmake --build build > nul

echo ========================================
echo Launching Particle GPU Integration Test...
echo ========================================

cd build\bin\Debug
.\particle_gpu_test.exe
pause
