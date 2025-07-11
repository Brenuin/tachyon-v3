
#include <GLFW/glfw3.h>
#include <chrono>
#include <thread>
#include "core/vec3.h"
#include "core/quaternion.h"
#include "core/matrix4.h"
#include "objects/rigid/box.h"
#include "render/render_utils.h"
#include <iostream>

using namespace tachyon;

const real DT = 0.016f;  // ~60 FPS

void testQuaternionRotation() {
    Vec3 v(1, 0, 0);
    Quaternion q(R_PI / 2, Vec3(0, 0, 1)); // 90° around Z-axis
    Vec3 rotated = q.rotate(v);
    std::cout << "Rotated Vec3: " << rotated << " (should be near (0,1,0))" << std::endl;
}

int main() {
    testQuaternionRotation();

    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(800, 800, "Box Rotation Test", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-10, 10, -10, 10, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Create a box
    Box box(Vec3(1.0f, 0.5f, 0.0f), 10.0f);
    box.pos = Vec3(0, 0, 0);
    box.vel = Vec3(0, 0, 0);
    box.angVel = Vec3(0, 1.0f, 1.0f);  // Rotate around Z-axis
    box.linearDamp = 1.0f;
    box.angularDamp = 1.0f;

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        // Simulate one step
        box.integrate(DT);

        // Draw rotated box
        drawBox3D(box.renderMatrix, box.getHalfExtents());

        // Display
        glfwSwapBuffers(window);
        glfwPollEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
    