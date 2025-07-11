#pragma once
#include "rigid_body.h"

namespace tachyon{
    class Box: public RigidBody{
public:
        Box(const Vec3& halfExtents, real mass);
        ShapeType getShapeType() const override { return ShapeType::Box; }
        const Vec3& getHalfExtents() const {return halfExtents;}

protected:
        Vec3 halfExtents;
};
}
