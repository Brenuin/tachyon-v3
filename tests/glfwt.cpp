#include <GLFW/glfw3.h>
#include <cmath>
#include <iostream>

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 600, "GLFWT - Visual Test", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    float posX = -1.0f;
    float speed = 0.01f;

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        posX += speed;
        if (posX > 1.0f || posX < -1.0f) speed *= -1.0f;

        glBegin(GL_TRIANGLES);
        glColor3f(0.2f, 0.8f, 0.4f);
        glVertex2f(posX, 0.0f);
        glVertex2f(posX - 0.1f, -0.1f);
        glVertex2f(posX + 0.1f, -0.1f);
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
