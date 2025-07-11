#pragma once
#include "rigid_body.h"

namespace tachyon {

class Sphere : public RigidBody {
public:
    ShapeType getShapeType() const override { return ShapeType::Sphere; }

    // Optional: constructor that sets radius, mass, etc.
    Sphere(real radius, real mass);
};

}
