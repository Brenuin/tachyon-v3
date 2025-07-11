#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "objects/particles/particle.h"

// Forward declare your launch function
void launchEulerIntegration(tachyon::particle::GPU* d_particles, int count, float dt);

int main() {
    using namespace tachyon;

    const float dt = 0.016f; // 60 FPS
    const int numParticles = 5;

    std::vector<particle> particles;
    std::vector<particle::GPU> gpuParticles;

    // === Create particles with varying states ===
    for (int i = 0; i < numParticles; ++i) {
        particle p;
        p.setPosition(0.0f, i * 2.0f, 0.0f);
        p.setVelocity(1.0f * i, 0.0f, 0.0f);
        p.setAcceleration(0.0f, -9.8f, 0.0f); // gravity
        p.setDamping(0.99f);
        p.setMass(1.0f); // 1kg
        p.updateGPUData();
        particles.push_back(p);
        gpuParticles.push_back(p.gpuData);
    }

    // === Allocate and copy to GPU ===
    particle::GPU* d_particles = nullptr;
    size_t bytes = sizeof(particle::GPU) * gpuParticles.size();

    cudaMalloc(&d_particles, bytes);
    cudaMemcpy(d_particles, gpuParticles.data(), bytes, cudaMemcpyHostToDevice);

    // === Launch integration kernel ===
    launchEulerIntegration(d_particles, numParticles, dt);
    cudaDeviceSynchronize();

    // === Copy back results ===
    cudaMemcpy(gpuParticles.data(), d_particles, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_particles);

    // === Print results ===
    for (int i = 0; i < numParticles; ++i) {
        std::cout << "Particle " << i << ":\n";
        std::cout << "  Position: (" << gpuParticles[i].pos.x << ", "
                                     << gpuParticles[i].pos.y << ", "
                                     << gpuParticles[i].pos.z << ")\n";
        std::cout << "  Velocity: (" << gpuParticles[i].vel.x << ", "
                                     << gpuParticles[i].vel.y << ", "
                                     << gpuParticles[i].vel.z << ")\n";
    }

    return 0;
}
