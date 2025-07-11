#include "objects/particles/particle_integrator.h"
#include <cmath>
#include <cassert>

namespace tachyon {

void EulerParticleIntegrator::integrate(particle& p, float dt) {
    assert(dt > 0.0f);
    if (p.getInverseMass() <= 0.0f) return;
    p.setPosition(p.getPosition() + p.getVelocity() * dt);

    Vec3 resultingAcc = p.getAcceleration();
    resultingAcc += p.forceAccum * p.getInverseMass();  

    p.setVelocity(p.getVelocity() + resultingAcc * dt);

    p.setVelocity(p.getVelocity() * std::pow(p.getDamping(), dt));

    p.clearAccumulator();
}

void RK4ParticleIntegrator::integrate(particle& p, float dt) {
    // TODO: implement RK4 integration
}

void VerletParticleIntegrator::integrate(particle& p, float dt) {
    assert(dt > 0.0f);
    if (!p.initialized) {
        p.previousPosition = p.position - p.velocity * dt;
        p.initialized = true;
    }

    Vec3 temp = p.position;
    p.position = p.position + (p.position - p.previousPosition) + p.acceleration * dt * dt;
    p.previousPosition = temp;
}

} // namespace tachyon
