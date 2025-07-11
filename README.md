✅ Core Dependencies
CUDA Toolkit (tested with CUDA 12.x or higher)
 – Includes nvcc, cuda_runtime.h, and device libraries.
 
C++17-compatible compiler
 – Tested with MSVC (Visual Studio 2022) and nvcc.

OpenGL development libraries
 – GLFW, GLAD, or other loaders linked correctly.

CMake (if using CMakeLists.txt, optional but helpful)

Recommended Development Environment
Windows 10/11 with:

Visual Studio 2022

CUDA Toolkit (set up for MSVC integration)

VS Code or Visual Studio IDE

NVIDIA GPU with Compute Capability 5.0+
 (e.g. GTX 10xx, RTX 20xx, 30xx, 40xx series)


🛠 Optional (but helpful)
Make (for Unix-like environments with compatible Makefile)

GLFW and GLAD source or binaries in external/ or properly installed

Environment variables or batch file (like run.bat) set up to compile with nvcc

You might also want to include a setup.md or BUILD.md to explain:
how to compile via make, nvcc, or .bat file

how to link against CUDA + OpenGL

where to put GLFW or GLAD (or use submodules)

expected file/folder structure (e.g., src/, include/, build/)
