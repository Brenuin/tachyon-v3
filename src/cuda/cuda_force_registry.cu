#include "cuda/cuda_force_registry.h"
#include "cuda/cuda_force_generator.h"

namespace cudaTachyon {

__global__ void applyForcesKernel(cudaParticle* particles, const ForceEntry* entries, int entryCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= entryCount) return;

    const ForceEntry& entry = entries[i];
    cudaParticle& p = particles[entry.particleIndex];

    switch (entry.type) {
        case GRAVITY:
            applyGravity(p, entry.vecParam);
            break;
        case DRAG:
            applyDrag(p, entry.param1);
            break;
        case SPRING:
            applySpring(p, particles[entry.otherIndex], entry.param2, entry.param1);
            break;
        case ANCHORED_SPRING:
            applyAnchoredSpring(p, entry.vecParam, entry.param2, entry.param1);
            break;
        case BUNGEE:
            applyBungee(p, particles[entry.otherIndex], entry.param2, entry.param1);
            break;
        case BUOYANCY:
            applyBuoyancy(p, entry.vecParam.y, entry.param2, entry.param1);
            break;
        case EXPLOSION:
            applyExplosion(p, entry.vecParam, entry.param2, entry.param1);
            break;
    }
}

void launchApplyForcesKernel(cudaParticle* d_particles, ForceEntry* d_entries, int entryCount) {
    int threads = 256;
    int blocks = (entryCount + threads - 1) / threads;
    applyForcesKernel<<<blocks, threads>>>(d_particles, d_entries, entryCount);
}

}
