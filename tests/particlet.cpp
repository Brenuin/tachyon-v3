#include <iostream>
#include <cassert>
#include "objects/particles/particle.h"
#include "objects/particles/particle_integrator.h"

using tachyon::Vec3;
using tachyon::real;
using tachyon::particle;
using tachyon::EulerParticleIntegrator;

const real EPSILON = 1e-5;

bool almost_equal(real a, real b, real epsilon = EPSILON) {
    return std::fabs(a - b) < epsilon;
}

void test_position_and_velocity() {
    particle p;
    p.setPosition(1.0f, 2.0f, 3.0f);
    p.setVelocity(4.0f, 5.0f, 6.0f);

    Vec3 pos = p.getPosition();
    Vec3 vel = p.getVelocity();

    assert(almost_equal(pos.x, 1.0f));
    assert(almost_equal(pos.y, 2.0f));
    assert(almost_equal(pos.z, 3.0f));

    assert(almost_equal(vel.x, 4.0f));
    assert(almost_equal(vel.y, 5.0f));
    assert(almost_equal(vel.z, 6.0f));

    std::cout << "Position and velocity setters/getters passed.\n";
}

void test_mass_damping() {
    particle p;
    p.setMass(2.0f);
    p.setDamping(0.9f);

    assert(almost_equal(p.getMass(), 2.0f));
    assert(almost_equal(p.getInverseMass(), 0.5f));
    assert(p.hasFiniteMass());
    assert(almost_equal(p.getDamping(), 0.9f));

    std::cout << "Mass, inverse mass, and damping passed.\n";
}

void test_force_application_and_clearing() {
    particle p;
    Vec3 force{10.0f, 0.0f, 0.0f};
    p.addForce(force);

    p.clearAccumulator();
    std::cout << "Force application and clearing passed.\n"; // if it crashes here, clearAccumulator failed
}

void test_integration() {
    particle p;
    p.setPosition(0.0f, 0.0f, 0.0f);
    p.setVelocity(1.0f, 0.0f, 0.0f);
    p.setAcceleration(0.0f, 1.0f, 0.0f);
    p.setMass(1.0f);
    p.setDamping(1.0f);

    EulerParticleIntegrator integrator;

    // First integration: velocity applied to position
    integrator.integrate(p, 1.0f);

    Vec3 pos1 = p.getPosition();
    Vec3 vel1 = p.getVelocity();

    assert(pos1.x > 0.0f);       
    assert(almost_equal(pos1.y, 0.0f));
    assert(almost_equal(vel1.y, 1.0f));

    
    integrator.integrate(p, 1.0f);

    Vec3 pos2 = p.getPosition();
    Vec3 vel2 = p.getVelocity();

    assert(pos2.y > 0.0f);       
    assert(vel2.y > 1.0f);       

    std::cout << "Euler integration with acceleration passed.\n";
}

int main() {
    test_position_and_velocity();
    test_mass_damping();
    test_force_application_and_clearing();
    test_integration();

    std::cout << "All particlet tests passed.\n";
    return 0;
}
