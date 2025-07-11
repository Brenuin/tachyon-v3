#include "world/world.h"
#include "render/render_utils.h"
#include <random>
#include <GLFW/glfw3.h>
#include "core/vec3.h"

namespace tachyon {

World::World(unsigned numSpheres, unsigned numBoxes)
    : grid(2.0f) // cell size
{
    std::mt19937 rng(44);
    std::uniform_real_distribution<real> dist(-10.0f, 10.0f);
    std::uniform_real_distribution<real> posRange(0.0f, 2.0f);
    // Add spheres
    for (unsigned i = 0; i < numSpheres; ++i) {
        real radius = posRange(rng);
        real mass = 20.0f;
        Sphere* s = new Sphere(radius, mass);
        s->pos = Vec3(dist(rng), dist(rng), 0.0f);
        s->vel = Vec3(1.5f * dist(rng), 1.5f * dist(rng), 0.0f);
        s->boundingRadius = radius;
        s->linearDamp = 0.98f;
        s->angularDamp = 1.0f;
        bodies.push_back(s);
    }

    // Add boxes
    for (unsigned i = 0; i < numBoxes; ++i) {
        Vec3 halfSize = Vec3(posRange(rng), posRange(rng), posRange(rng));
        real mass = 20.0f;
        Box* b = new Box(halfSize, mass);
        b->pos = Vec3(dist(rng), dist(rng), 0.0f);
        b->vel = Vec3(1.5f * dist(rng), 1.5f * dist(rng), 0.0f);
        b->angVel = Vec3(0, 0, 2.0f); 
        b->boundingRadius = halfSize.magnitude();
        b->linearDamp = 0.98f;
        b->angularDamp = 0.98f;
        bodies.push_back(b);
    }
}

World::~World() {
    for (RigidBody* body : bodies) delete body;
}

void World::applyForces(real dt) {
    grid.clear();
    for (RigidBody* body : bodies) {
        body->addForce(gravity * (1.0f / body->inverseMass));
        body->integrate(dt);
        grid.insert(body);
    }
}

void World::handleBounds() {
    for (RigidBody* body : bodies) {
        if (body->pos.x < -worldBound) {
            body->pos.x = -worldBound;
            body->vel.x *= -0.7f;
        }
        if (body->pos.x > worldBound) {
            body->pos.x = worldBound;
            body->vel.x *= -0.7f;
        }
        if (body->pos.y < -worldBound) {
            body->pos.y = -worldBound;
            body->vel.y *= -0.7f;
        }
        if (body->pos.y > worldBound) {
            body->pos.y = worldBound;
            body->vel.y *= -0.7f;
        }
    }
}

void World::update(real dt) {
    applyForces(dt);
    handleBounds();

    auto pairs = grid.computePotentialPairs();
    auto contacts = generator.generateContacts(pairs);
    resolver.resolveContactsIterative(contacts, dt, 50);
}

void World::render() {
    drawGrid3D(1.0f, 20);
    for (RigidBody* body : bodies) {
        if (auto* s = dynamic_cast<Sphere*>(body)) {
            drawSphere3D(s->pos, s->boundingRadius);
        } else if (auto* b = dynamic_cast<Box*>(body)) {
            drawBox3D(b->renderMatrix, b->getHalfExtents());
        }
    }
}
}
