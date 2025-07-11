cmake_minimum_required(VERSION 3.22)
project(tachyon_v2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# GLFW include and lib paths
include_directories("C:/Libraries/glfw-3.4.bin.WIN64/include")
link_directories("C:/Libraries/glfw-3.4.bin.WIN64/lib-vc2022")

# CUDA include path for IntelliSense and C++ compilation
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include")

# ======== MAIN RUNTIME DEMO (RE-ENABLED) ========
add_executable(main
    src/main/main.cpp
    src/objects/rigid/sphere.cpp
    src/objects/rigid/box.cpp
    src/objects/rigid/rigid_body.cpp
    src/collision/broadphase/grid.cpp
    src/collision/narrowphase/contact_generator.cpp
    src/collision/narrowphase/contact_resolver.cpp
    src/render/render_utils.cpp
    src/world/world.cpp 
)
target_link_libraries(main glfw3 opengl32 gdi32)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include)
# ======== BOX ROTATION VISUAL TEST ========
add_executable(box_rotation_test
    tests/box_rotation_test.cpp
    src/objects/rigid/box.cpp
    src/objects/rigid/rigid_body.cpp
    src/render/render_utils.cpp
)
target_link_libraries(box_rotation_test glfw3 opengl32 gdi32)
target_include_directories(box_rotation_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ======== CONTACT RESOLVER TEST ========
add_executable(contact_test
    tests/contact_test.cpp
    src/objects/rigid/sphere.cpp
    src/objects/rigid/rigid_body.cpp
    src/collision/narrowphase/contact_resolver.cpp
    src/collision/narrowphase/contact_generator.cpp
)
target_include_directories(contact_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ======== INTEGRATOR TEST ========
add_executable(particle_gpu_test
    tests/particle_gpu_test.cpp
    src/objects/particles/particle.cpp
    src/objects/particles/particle_integrator.cpp
    src/cuda/particle_integrator.cu
)
target_include_directories(particle_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(particle_gpu_test cuda)

# ======== VISUAL TEST ========
add_executable(glfwt tests/glfwt.cpp)
target_link_libraries(glfwt glfw3 opengl32 gdi32)
target_include_directories(glfwt PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ======== VECTOR UNIT TEST ========
add_executable(vectort tests/vectort.cpp)
target_include_directories(vectort PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ======== MAIN TEST RUNNER ========
add_executable(main_test tests/main_test.cpp)
target_include_directories(main_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ======== PARTICLE TEST ========
add_executable(particlet
    tests/particlet.cpp
    src/objects/particles/particle.cpp
    src/objects/particles/particle_integrator.cpp
)
target_include_directories(particlet PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ======== CUDA PARTICLE TEST ========
add_executable(cuda_particle_test
    tests/cuda_particle_test.cpp
    src/world/cuda_world.cu
    src/cuda/cuda_force_registry.cu
)
target_include_directories(cuda_particle_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cuda_particle_test cuda glfw3 opengl32 gdi32)
