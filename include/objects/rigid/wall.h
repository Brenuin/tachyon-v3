#include "rigid/box.h"
#include "core/vec3.h"

tachyon::Box* createWall(const tachyon::Vec3& position, const tachyon::Vec3& halfExtents) {
    using namespace tachyon;

    Box* wall = new Box(halfExtents, 0.0f);  // Zero mass → infinite mass (static)
    wall->pos = position;
    wall->vel = Vec3(0, 0, 0);
    wall->orient = Quaternion::identity();
    wall->setMass(0.0f);  // Marks it static
    wall->setBoundingRadius(halfExtents.magnitude()); // Optional
    wall->setInertiaTensor(Matrix3::zero()); // Prevents rotation
    wall->calculateDerivedData();
    return wall;
}
