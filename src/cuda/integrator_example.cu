// src/cuda/particle_integrator.cu
#include <cuda_runtime.h>
#include "objects/particles/particle.h"

__global__ void integrateEulerKernel(tachyon::particle::GPU* particles, int count, float dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= count) return;

    auto& p = particles[i];

    if (p.invMass <= 0.0f) return;

    float3 resultingAcc = p.acc;
    resultingAcc.x += p.forceAccum.x * p.invMass;
    resultingAcc.y += p.forceAccum.y * p.invMass;
    resultingAcc.z += p.forceAccum.z * p.invMass;

    p.vel.x += resultingAcc.x * dt;
    p.vel.y += resultingAcc.y * dt;
    p.vel.z += resultingAcc.z * dt;

    p.pos.x += p.vel.x * dt;
    p.pos.y += p.vel.y * dt;
    p.pos.z += p.vel.z * dt;

    float dampingFactor = powf(0.99f, dt); // hardcoded or pass in
    p.vel.x *= dampingFactor;
    p.vel.y *= dampingFactor;
    p.vel.z *= dampingFactor;

    p.forceAccum = make_float3(0, 0, 0); // clear accumulator
}

// Launch function
void launchEulerIntegration(tachyon::particle::GPU* d_particles, int count, float dt) {
    int blockSize = 256;
    int numBlocks = (count + blockSize - 1) / blockSize;
    integrateEulerKernel<<<numBlocks, blockSize>>>(d_particles, count, dt);
}

std::vector<tachyon::particle> particles = ...; // init with values

// Update GPU structs
for (auto& p : particles) {
    p.updateGPUData();
}

// Copy to GPU
tachyon::particle::GPU* d_particles;
cudaMalloc(&d_particles, sizeof(tachyon::particle::GPU) * particles.size());
cudaMemcpy(d_particles, particles.data(), sizeof(tachyon::particle::GPU) * particles.size(), cudaMemcpyHostToDevice);

// Run integration
launchEulerIntegration(d_particles, particles.size(), timestep);

// Copy back (optional if you want to sync)
cudaMemcpy(particles.data(), d_particles, sizeof(tachyon::particle::GPU) * particles.size(), cudaMemcpyDeviceToHost);

cudaFree(d_particles);
