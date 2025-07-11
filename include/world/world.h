#pragma once
#include <vector>
#include "objects/rigid/rigid_body.h"
#include "objects/rigid/sphere.h"
#include "objects/rigid/box.h"
#include "collision/broadphase/grid.h"
#include "collision/narrowphase/contact_generator.h"
#include "collision/narrowphase/contact_resolver.h"

namespace tachyon {

class World {
public:
    World(unsigned numSpheres = 10, unsigned numBoxes = 10);
    ~World();

    void update(real dt);
    void render();

private:
    std::vector<RigidBody*> bodies; 
    BroadphaseGrid grid;
    ContactGenerator generator;
    ContactResolver resolver;

    const real worldBound = 10.0f;
    const Vec3 gravity = Vec3(0.0f, -9.81f, 0.0f);

    void applyForces(real dt);
    void handleBounds();
};

}

