// src/world/cuda_world.cu
#include <cuda_runtime.h>
#include "cuda/cuda_particle.h"
#include "cuda/cuda_force_registry.h"
#include "world/cuda_world.h"

namespace cudaTachyon {

    __global__ void integrateParticlesKernel(cudaParticle* particles, int count, float dt) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        particles[i].integrate(dt);
    }

    void runCudaWorld(std::vector<cudaParticle>& particles, std::vector<ForceEntry>& registry, float dt) {
        // Allocate GPU memory
        cudaParticle* d_particles = nullptr;
        ForceEntry* d_registry = nullptr;

        cudaMalloc(&d_particles, particles.size() * sizeof(cudaParticle));
        cudaMemcpy(d_particles, particles.data(), particles.size() * sizeof(cudaParticle), cudaMemcpyHostToDevice);

        cudaMalloc(&d_registry, registry.size() * sizeof(ForceEntry));
        cudaMemcpy(d_registry, registry.data(), registry.size() * sizeof(ForceEntry), cudaMemcpyHostToDevice);

        // Apply forces
        launchApplyForcesKernel(d_particles, d_registry, static_cast<int>(registry.size()));
        cudaDeviceSynchronize();

        // Integrate
        int threads = 256;
        int blocks = (particles.size() + threads - 1) / threads;
        integrateParticlesKernel<<<blocks, threads>>>(d_particles, static_cast<int>(particles.size()), dt);
        cudaDeviceSynchronize();

        // Copy back to CPU
        cudaMemcpy(particles.data(), d_particles, particles.size() * sizeof(cudaParticle), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_particles);
        cudaFree(d_registry);
    }

}
