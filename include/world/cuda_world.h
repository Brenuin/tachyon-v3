// include/world/cuda_world.h
#pragma once
#include <vector>
#include "cuda/cuda_particle.h"
#include "cuda/cuda_force_registry.h"

namespace cudaTachyon {
    void runCudaWorld(std::vector<cudaParticle>& particles, std::vector<ForceEntry>& registry, float dt);
}
