#pragma once
#include <math.h>
#include <cuda_runtime.h>

namespace cudaTachyon{

    struct cudaParticle
    {
        float3 pos;
        float3 vel;
        float3 acc;
        float3 force;

        float invMass;
        float damping;
        float radius;

        float3 prevPos;
        uint8_t initialized;

        __host__ __device__ 
        cudaParticle(){
            pos = vel = acc = force = make_float3(0.0f,0.0f,0.0f);
            prevPos = make_float3(0.0f,0.0f,0.0f);
            invMass = 1.0f;
            damping = 0.99f;
            radius = 1.0f;
            initialized = 0;
        }



        __host__ __device__ inline void clearForce(){
            force = make_float3(0.0f,0.0f,0.0f);
        }
        __host__ __device__ inline void addForce(const float3& f){
            force.x += f.x;
            force.y += f.y;
            force.z += f.z;

        }
        __host__ __device__ inline bool hasFiniteMass() const{
            return invMass > 0.0f;
        }

        __host__ __device__ inline void integrate(float dt){
            if (!hasFiniteMass()) return;
            float3 resultingAcc = acc;
            resultingAcc.x += force.x * invMass;
            resultingAcc.y += force.y * invMass;
            resultingAcc.z += force.z * invMass;

            vel.x += resultingAcc.x*dt;
            vel.y += resultingAcc.y*dt;
            vel.z += resultingAcc.z*dt;

            vel.x *= powf(damping,dt);
            vel.y *= powf(damping,dt);
            vel.z *= powf(damping,dt);

            prevPos = pos;

            pos.x += vel.x*dt;
            pos.y += vel.y*dt;
            pos.z += vel.z*dt;

            clearForce();

        }
    };
    
}