#pragma once
#include <cuda_runtime.h>
#include <math.h>


namespace cudaTachyon{


    __host__ __device__ inline float3 add(const float3& a, const float3& b){
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
    __host__ __device__ inline float3 sub(const float3& a, const float3& b){
        return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
    }
    __host__ __device__ inline float3 scale(const float3& v, float s){
        return make_float3(v.x*s, v.y*s, v.z*s);
    }
    __host__ __device__ inline float dot(const float3& a, const float3&b){
        return (a.x*b.x + a.y*b.y + a.z*b.z);
    }
    __host__ __device__ inline float lengthSq(const float3& v){
        return dot(v,v);
    }
    __host__ __device__ inline float length(const float3& v) {
        return sqrtf(lengthSq(v));
    }

    __host__ __device__ inline float3 normalize(const float3& v) {
        float len = length(v);
        if(len > 1e-6f){
            return scale(v,1.0f/len);
        }else{
            return make_float3(0.0f,0.0f,0.0f);
        }
    }

    __host__ __device__ inline float3 cross(const float3& a, const float3& b) {
        return make_float3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    __host__ __device__ inline float3 clamp(const float3& v, float minVal, float maxVal) {
        return make_float3(
            fminf(fmaxf(v.x, minVal), maxVal),
            fminf(fmaxf(v.y, minVal), maxVal),
            fminf(fmaxf(v.z, minVal), maxVal)
        );
    }

    __host__ __device__ inline float3 lerp(const float3& a, const float3& b, float t) {
        return add(scale(a, 1.0f - t), scale(b, t));
    }

}