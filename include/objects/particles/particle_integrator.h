#pragma once
#include "particle.h"
namespace tachyon {

class ParticleIntegrator {
public:
    virtual void integrate(particle& p, float dt) = 0;
    virtual ~ParticleIntegrator() {}
};

class EulerParticleIntegrator : public ParticleIntegrator {
public:
    void integrate(particle& p, float dt) override;
};

class RK4ParticleIntegrator : public ParticleIntegrator {
public:
    void integrate(particle& p, float dt) override;
};

class VerletParticleIntegrator : public ParticleIntegrator {
public:
    void integrate(particle& p, float dt) override;

private:
    Vec3 previousPosition = {0, 0, 0}; // optional, used if needed
};

} // namespace tachyon
