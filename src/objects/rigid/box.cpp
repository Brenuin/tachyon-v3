#include "objects/rigid/box.h"

namespace tachyon {

Box::Box(const Vec3& halfExtents, real mass)
    : halfExtents(halfExtents)
{
    boundingRadius = halfExtents.magnitude(); // Approximate for sphere-bound collisions
    setMass(mass);

    // Moment of inertia tensor for a solid box:
    // Ixx = (1/12) * m * (h^2 + d^2), and so on
    real m = 1.0f / inverseMass;
    real x2 = 4.0f * halfExtents.x * halfExtents.x;
    real y2 = 4.0f * halfExtents.y * halfExtents.y;
    real z2 = 4.0f * halfExtents.z * halfExtents.z;

    real Ixx = (1.0f / 12.0f) * m * (y2 + z2);
    real Iyy = (1.0f / 12.0f) * m * (x2 + z2);
    real Izz = (1.0f / 12.0f) * m * (x2 + y2);

    setInertiaTensor(Matrix3(
        Ixx, 0,   0,
        0,   Iyy, 0,
        0,   0,   Izz
    ));
}

}
