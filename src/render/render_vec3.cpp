#include <Windows.h>
#include "render/render_vec3.h"
#include <stdexcept>
#include <gl/GL.h>

namespace tachyon {

RenderVec3::RenderVec3(int w, int h, const char* title) : width(w), height(h) {
    if (!glfwInit()) throw std::runtime_error("Failed to initialize GLFW");

    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window);
    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-1, 1, -1, 1, 2, 100); // Perspective projection
    glMatrixMode(GL_MODELVIEW);
}

RenderVec3::~RenderVec3() {
    glfwDestroyWindow(window);
    glfwTerminate();
}

bool RenderVec3::shouldClose() const {
    return glfwWindowShouldClose(window);
}

void RenderVec3::beginFrame() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(0, 0, -20);
    glRotatef(30, 1, 0, 0);
    glRotatef(30, 0, 1, 0);
}

void RenderVec3::endFrame() {
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void RenderVec3::drawAxis(float length) {
    glLineWidth(2.0f); // make axes stand out

    glBegin(GL_LINES);
    // X axis - Red
    glColor3f(1, 0, 0);
    glVertex3f(-length, 0, 0);
    glVertex3f(length, 0, 0);

    // Y axis - Green
    glColor3f(0, 1, 0);
    glVertex3f(0, -length, 0);
    glVertex3f(0, length, 0);

    // Z axis - Blue
    glColor3f(0, 0, 1);
    glVertex3f(0, 0, -length);
    glVertex3f(0, 0, length);
    glEnd();

    // Draw notches every 1.0 unit
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    glColor3f(0.3f, 0.3f, 0.3f); // dim gray for notches

    float spacing = 1.0f;
    float notchSize = 0.1f;

    for (float i = -length; i <= length; i += spacing) {
        if (std::abs(i) < 1e-5f) continue; // skip origin

        // Notches along X
        glVertex3f(i, -notchSize, 0);
        glVertex3f(i, notchSize, 0);

        // Notches along Y
        glVertex3f(-notchSize, i, 0);
        glVertex3f(notchSize, i, 0);

        // Notches along Z
        glVertex3f(-notchSize, 0, i);
        glVertex3f(notchSize, 0, i);
    }
    glEnd();
}

void RenderVec3::drawVector(const Vec3& origin, const Vec3& direction, const Vec3& color) {
    glColor3f(color.x, color.y, color.z);
    glBegin(GL_LINES);
    glVertex3f(origin.x, origin.y, origin.z);
    glVertex3f(origin.x + direction.x, origin.y + direction.y, origin.z + direction.z);
    glEnd();
}

void RenderVec3::drawPoint(const Vec3& position, const Vec3& color, float size) {
    glColor3f(color.x, color.y, color.z);
    glPointSize(size);
    glBegin(GL_POINTS);
    glVertex3f(position.x, position.y, position.z);
    glEnd();
}

} // namespace tachyon
