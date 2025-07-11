#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <iostream>
#include <vector>
#include "cuda/cuda_particle.h"
#include "cuda/cuda_force_registry.h"
#include "world/cuda_world.h"

using namespace cudaTachyon;

void renderParticles(const std::vector<cudaParticle>& particles, float pointSize) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    glPointSize(pointSize);
    glBegin(GL_POINTS);
    for (const auto& p : particles) {
        glVertex3f(p.pos.x, p.pos.y, p.pos.z);
    }
    glEnd();
}

int main() {
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(800, 600, "CUDA Particles", NULL, NULL);
    if (!window) return -1;
    glfwMakeContextCurrent(window);

    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-10.0f, 10.0f, -10.0f, 30.0f, -1.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);

    std::vector<cudaParticle> particles(2);
    particles[0].pos = make_float3(0.0f, 10.0f, 0.0f);
    particles[1].pos = make_float3(1.0f, 20.0f, 0.0f);

    std::vector<ForceEntry> registry;
    float3 gravity = make_float3(0.0f, -9.81f, 0.0f);
    registry.push_back({0, -1, GRAVITY, gravity, 0.0f, 0.0f});
    registry.push_back({1, -1, GRAVITY, gravity, 0.0f, 0.0f});

    float dt = 0.01f;

    while (!glfwWindowShouldClose(window)) {
        runCudaWorld(particles, registry, dt);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        renderParticles(particles, 8.0f);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
