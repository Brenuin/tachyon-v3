#include <GLFW/glfw3.h>
#include <chrono>
#include <iostream>
#include "world/world.h"
#include <thread>

using namespace tachyon;
const real DT = 0.008f;

int main() {
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(800, 800, "Tachyon World", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glEnable(GL_DEPTH_TEST);

    // Perspective camera setup
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-1, 1, -1, 1, 2.5, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -30.0f);
    glRotatef(30, 1.0f, 0.0f, 0.0f);
    glRotatef(20, 0.0f, 1.0f, 0.0f);

    World world(10,10);

    // Performance tracking
    int frameCount = 0;
    double totalTimeMs = 0.0;

    while (!glfwWindowShouldClose(window)) {
        auto start = std::chrono::high_resolution_clock::now();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        world.update(DT);
        world.render();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        totalTimeMs += elapsed.count();
        frameCount++;

        // Print average every 100 frames
        if (frameCount % 100 == 0) {
            double avgMs = totalTimeMs / frameCount;
            double avgFPS = 1000.0 / avgMs;
            std::cout << "Average Frame Time: " << avgMs << " ms, FPS: " << avgFPS << "\n";
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
