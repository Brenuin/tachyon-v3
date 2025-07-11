#pragma once
#include <cuda_runtime.h>
#include "cuda_particle.h"

namespace cudaTachyon {
    enum ForceType : uint8_t {
        GRAVITY = 0,
        DRAG = 1,
        SPRING = 2,
        ANCHORED_SPRING = 3,
        BUNGEE = 4,
        BUOYANCY = 5,
        AERO = 6,
        EXPLOSION = 7
    };


    struct ForceEntry {
        int particleIndex;
        int otherIndex;
        ForceType type;
        float3 vecParam;
        float param1;
        float param2;
    };

    void launchApplyForcesKernel(cudaParticle* d_particles, ForceEntry* d_entries, int entryCount);
}
