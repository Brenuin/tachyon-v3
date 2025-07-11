## Requirements

To build and run **Tachyon V3**, make sure the following are installed:

---

### Core Dependencies

- **CUDA Toolkit** (tested with CUDA 12.x or higher)  
  Includes `nvcc`, `cuda_runtime.h`, and device libraries.

- **C++17-compatible compiler**  
  Tested with **MSVC (Visual Studio 2022)** and `nvcc`.

- **OpenGL development libraries**  
  GLFW, GLAD, or other OpenGL loaders properly linked.

- **CMake** *(optional but helpful)*  
  Only needed if using `CMakeLists.txt`.

---

### Recommended Development Environment

- Windows 10/11
- Visual Studio 2022 with Desktop C++ + CUDA support
- VS Code or Visual Studio IDE
- NVIDIA GPU (Compute Capability 5.0+)  
  *e.g. GTX 10xx, RTX 20xx, 30xx, 40xx series*

---

### Optional (But Helpful)

- `make` (for Unix-like environments using the Makefile)
- GLFW and GLAD source or binaries in `/external/`  
- Environment setup via a batch file like `run.bat`  
  (sets up `nvcc`, MSVC, and includes correctly)

