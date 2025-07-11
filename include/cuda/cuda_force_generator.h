#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "cuda_particle.h"
#include "utilities/float3_utils.h"

namespace cudaTachyon {

    __host__ __device__ inline void applyGravity(cudaParticle& p, const float3& gravity) {
        if (p.hasFiniteMass()) {
            float3 force = scale(gravity, 1.0f / p.invMass);
            p.addForce(force);
        }
    }

    __host__ __device__ inline void applyDrag(cudaParticle& p, float dragCoeff) {
        if (p.hasFiniteMass()) {
            float3 drag = scale(p.vel, -dragCoeff);
            p.addForce(drag);
        }
    }

    __host__ __device__ inline void applySpring(cudaParticle& p1, const cudaParticle& p2, float restLength, float stiffness) {
        float3 delta = sub(p1.pos, p2.pos);
        float dist = length(delta);
        float magnitude = -stiffness * (dist - restLength);
        float3 force = scale(normalize(delta), magnitude);
        p1.addForce(force);
    }

    __host__ __device__ inline void applyAnchoredSpring(cudaParticle& p, const float3& anchor, float restLength, float stiffness) {
        float3 delta = sub(p.pos, anchor);
        float dist = length(delta);
        float magnitude = -stiffness * (dist - restLength);
        float3 force = scale(normalize(delta), magnitude);
        p.addForce(force);
    }

    __host__ __device__ inline void applyBungee(cudaParticle& p, const cudaParticle& anchor, float restLength, float stiffness) {
        float3 delta = sub(p.pos, anchor.pos);
        float dist = length(delta);
        if (dist <= restLength) return;
        float magnitude = -stiffness * (dist - restLength);
        float3 force = scale(normalize(delta), magnitude);
        p.addForce(force);
    }

    __host__ __device__ inline void applyBuoyancy(cudaParticle& p, float fluidHeight, float maxDepth, float volume, float liquidDensity = 1000.0f) {
        float depth = p.pos.y;
        if (depth >= fluidHeight + maxDepth) return; // Fully out of fluid
        float buoyantForce = 0.0f;

        if (depth <= fluidHeight - maxDepth) {
            buoyantForce = liquidDensity * volume;
        } else {
            float submersion = (fluidHeight - depth + maxDepth) / (2.0f * maxDepth);
            buoyantForce = liquidDensity * volume * submersion;
        }

        p.addForce(make_float3(0.0f, buoyantForce, 0.0f));
    }

    __host__ __device__ inline void applyExplosion(cudaParticle& p, const float3& center, float radius, float strength) {
        float3 offset = sub(p.pos, center);
        float distSq = lengthSq(offset);
        if (distSq > radius * radius) return;
        float dist = sqrtf(distSq);
        float magnitude = strength * (1.0f - (dist / radius));
        float3 force = scale(normalize(offset), magnitude);
        p.addForce(force);
    }

    __host__ __device__ inline void applyPairwiseGravity(cudaParticle& a, const cudaParticle& b, float G) {
        float3 offset = sub(b.pos, a.pos);
        float distSq = lengthSq(offset) + 1e-6f;
        float dist = sqrtf(distSq);
        float3 dir = scale(offset, 1.0f / dist);
        float forceMag = G / distSq;
        float3 force = scale(dir, forceMag / (a.invMass * b.invMass));
        if (a.hasFiniteMass()) {
            a.addForce(force);
        }
    }
}
