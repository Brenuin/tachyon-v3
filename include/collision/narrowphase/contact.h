#pragma once
#include "objects/rigid/rigid_body.h"
#include "core/vec3.h"
#include "utilities/precision.h"


namespace tachyon{
    struct Contact{
        RigidBody* bodyA = nullptr;
        RigidBody* bodyB = nullptr;
        Vec3 contactPoint;
        Vec3 contactNormal;
        real penetration = 0.0;
        real restitution = 0.2f;
        real friction = 0.0f;
        unsigned resolveCount = 0;
    };
}