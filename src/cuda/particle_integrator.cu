#include <cuda_runtime.h>
#include "objects/particles/particle.h"

__global__ void integrateEulerKernel(tachyon::particle::GPU* particles, int count, float dt){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=  count) return;
    auto& p = particles[i];
    if(p.invMass<=0.0f) return;
    float3 resultingAcc = p.acc;
    //fac = force accumulated
    resultingAcc.x += p.fac.x * p.invMass;
    resultingAcc.y += p.fac.y * p.invMass;
    resultingAcc.z += p.fac.z * p.invMass;

    p.vel.x += resultingAcc.x * dt;
    p.vel.y += resultingAcc.y * dt;
    p.vel.z += resultingAcc.z * dt;

    p.vel.x *= powf(p.damp,dt);
    p.vel.y *= powf(p.damp,dt);
    p.vel.z *= powf(p.damp,dt);

    p.pos.x += p.vel.x * dt;
    p.pos.y += p.vel.y * dt;
    p.pos.z += p.vel.z * dt;

    p.fac = make_float3(0.0f, 0.0f, 0.0f);


}

void launchEulerIntegration(tachyon::particle::GPU* d_particles, int count, float dt){
    int blockSize = 256;
    int numBlocks = (count + blockSize -1 )/blockSize;
    integrateEulerKernel<<<numBlocks,blockSize>>>(d_particles,count,dt);
}