@echo off
echo [CUDA PARTICLE TEST] Building cuda_particle_test...

REM Navigate to build directory
cd /d %~dp0\..\build

REM Build only the cuda_particle_test target using MSBuild
msbuild.exe tachyon_v2.sln /t:cuda_particle_test /p:Configuration=Debug /nologo

REM Check for success
IF EXIST bin\Debug\cuda_particle_test.exe (
    echo [SUCCESS] Build complete. Running cuda_particle_test...
    bin\Debug\cuda_particle_test.exe
) ELSE (
    echo [ERROR] cuda_particle_test.exe not found. Build may have failed.
)

pause
