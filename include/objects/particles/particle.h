#pragma once
#include "core/vec3.h"


#include <cuda_runtime.h>

namespace tachyon{

    class particle{
        public:
            Vec3 position;
            Vec3 velocity;
            Vec3 acceleration;
            Vec3 forceAccum;

            real inverseMass;
            real damping;
            real radius;

            Vec3 previousPosition;
            bool initialized;

            struct GPU
            {
                float3 pos, vel, acc, fac;
                float invMass;
                float damp;
                /* data */
            }gpuData;
            

            particle();

            void updateGPUData();
            void integrate(real dt);
            void clearAccumulator();
            void addForce(const Vec3 force);

            void setPosition(const Vec3& p);
            void setPosition(real x, real y, real z);
            Vec3 getPosition() const;

            void setVelocity(const Vec3& v);
            void setVelocity(real x, real y, real z);
            Vec3 getVelocity() const;

            void setAcceleration(const Vec3& a);
            void setAcceleration(real x, real y, real z);
            Vec3 getAcceleration() const;

            void setDamping(real d);
            real getDamping() const;

            void setMass(real mass);
            real getMass() const;
            real getInverseMass() const;
            bool hasFiniteMass() const;


    };


}