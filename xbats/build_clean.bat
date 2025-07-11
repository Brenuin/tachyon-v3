@echo off
cd ..
echo Deleting IntelliSense cache...
del /q .vscode\*.browse.VC.db
rmdir /s /q .vscode\ipch
cmake -S . -B build
cmake --build build
