@echo off
cd ..
setlocal enabledelayedexpansion

echo ===============================
echo 🔍 Printing Selected Files
echo ===============================

:: ==== FILE FLAGS ====
set PRINT_main=false
set PRINT_runbat=false
set PRINT_cmake=true

:: ==== FOLDER FLAGS ====
set PRINT_folder_core=true
set PRINT_folder_collision=false
set PRINT_folder_cuda=true
set PRINT_folder_particles=true
set PRINT_folder_render=false
set PRINT_folder_rigid=false
set PRINT_folder_spatial=false
set PRINT_folder_systems=false
set PRINT_folder_world=false
set PRINT_folder_constraint=false

:: ==== OBJECT SUBFOLDERS ====
set PRINT_folder_objects=true
set PRINT_folder_objects_cuda=true
set PRINT_folder_objects_particles=true
set PRINT_folder_objects_rigid=true

:: ==== Root Files ====
echo [DEBUG] Checking root file flags...

if "%PRINT_main%"=="true" (
    echo === [src/main/main.cpp] ===
    type src\main\main.cpp
    echo.
)

if "%PRINT_runbat%"=="true" (
    echo === [run.bat] ===
    type run.bat
    echo.
)

if "%PRINT_cmake%"=="true" (
    echo === [CMakeLists.txt] ===
    type CMakeLists.txt
    echo.
)

:: ==== INCLUDE FOLDERS ====
if "%PRINT_folder_core%"=="true" (
    call :PrintFolder include\core "include/core"
)
if "%PRINT_folder_collision%"=="true" (
    call :PrintFolder include\collision "include/collision"
)
if "%PRINT_folder_cuda%"=="true" (
    call :PrintFolder include\cuda "include/cuda"
)
if "%PRINT_folder_particles%"=="true" (
    call :PrintFolder include\objects\particles "include/objects/particles"
)
if "%PRINT_folder_render%"=="true" (
    call :PrintFolder include\render "include/render"
)
if "%PRINT_folder_rigid%"=="true" (
    call :PrintFolder include\objects\rigid "include/objects/rigid"
)
if "%PRINT_folder_spatial%"=="true" (
    call :PrintFolder include\spatial "include/spatial"
)
if "%PRINT_folder_systems%"=="true" (
    call :PrintFolder include\systems "include/systems"
)
if "%PRINT_folder_world%"=="true" (
    call :PrintFolder include\world "include/world"
)
if "%PRINT_folder_constraint%"=="true" (
    call :PrintFolder include\constraint "include/constraint"
)

:: ==== OBJECT INCLUDE SUBFOLDERS ====
if "%PRINT_folder_objects%"=="true" (
    if "%PRINT_folder_objects_cuda%"=="true" (
        call :PrintFolder include\objects\cudaObjects "include/objects/cudaObjects"
    )
    if "%PRINT_folder_objects_particles%"=="true" (
        call :PrintFolder include\objects\particles "include/objects/particles"
    )
    if "%PRINT_folder_objects_rigid%"=="true" (
        call :PrintFolder include\objects\rigid "include/objects/rigid"
    )
)

:: ==== SRC FOLDERS ====
if "%PRINT_folder_core%"=="true" (
    call :PrintFolder src\core "src/core"
)
if "%PRINT_folder_collision%"=="true" (
    call :PrintFolder src\collision "src/collision"
)
if "%PRINT_folder_cuda%"=="true" (
    call :PrintFolder src\cuda "src/cuda"
)
if "%PRINT_folder_particles%"=="true" (
    call :PrintFolder src\objects\particles "src/objects/particles"
)
if "%PRINT_folder_render%"=="true" (
    call :PrintFolder src\render "src/render"
)
if "%PRINT_folder_rigid%"=="true" (
    call :PrintFolder src\objects\rigid "src/objects/rigid"
)
if "%PRINT_folder_spatial%"=="true" (
    call :PrintFolder src\spatial "src/spatial"
)
if "%PRINT_folder_systems%"=="true" (
    call :PrintFolder src\systems "src/systems"
)
if "%PRINT_folder_world%"=="true" (
    call :PrintFolder src\world "src/world"
)
if "%PRINT_folder_constraint%"=="true" (
    call :PrintFolder src\constraint "src/constraint"
)

:: ==== OBJECT SRC SUBFOLDERS ====
if "%PRINT_folder_objects%"=="true" (
    if "%PRINT_folder_objects_cuda%"=="true" (
        call :PrintFolder src\objects\cudaObjects "src/objects/cudaObjects"
    )
    if "%PRINT_folder_objects_particles%"=="true" (
        call :PrintFolder src\objects\particles "src/objects/particles"
    )
    if "%PRINT_folder_objects_rigid%"=="true" (
        call :PrintFolder src\objects\rigid "src/objects/rigid"
    )
)

echo.
echo ✅ Done printing files.
pause
goto :eof

:: ==== Helper Function ====
:PrintFolder
set "TARGETFOLDER=%~1"
set "PRINTLABEL=%~2"
echo.
echo [DEBUG] Entering :PrintFolder for "%PRINTLABEL%" at "%TARGETFOLDER%"
echo === [%PRINTLABEL%] ===

set "FOUND=false"

for /f "delims=" %%F in ('dir /b "%TARGETFOLDER%\*.h" 2^>nul') do (
    echo --- %TARGETFOLDER%\%%F
    type "%TARGETFOLDER%\%%F"
    echo.
    set "FOUND=true"
)

for /f "delims=" %%F in ('dir /b "%TARGETFOLDER%\*.cpp" 2^>nul') do (
    echo --- %TARGETFOLDER%\%%F
    type "%TARGETFOLDER%\%%F"
    echo.
    set "FOUND=true"
)

for /f "delims=" %%F in ('dir /b "%TARGETFOLDER%\*.cu" 2^>nul') do (
    echo --- %TARGETFOLDER%\%%F
    type "%TARGETFOLDER%\%%F"
    echo.
    set "FOUND=true"
)

if "%FOUND%"=="false" (
    echo [DEBUG] No .h/.cpp/.cu files found in "%TARGETFOLDER%"
)
goto :eof

