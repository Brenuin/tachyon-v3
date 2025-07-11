#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "utilities/float3_utils.h" 

namespace cudaTachyon {

    struct CudaDynamicObject {
        float3 pos;
        float radius;

        __host__ __device__
        CudaDynamicObject() {
            pos = make_float3(0.0f, 0.0f, 0.0f);
            radius = 1.0f;
        }
    };

}
