#include "objects/rigid/sphere.h"

namespace tachyon {

Sphere::Sphere(real radius, real mass) {
    boundingRadius = radius;
    setMass(mass);

    // Moment of inertia for a solid sphere: I = 2/5 * m * r^2
    real I = (2.0f / 5.0f) * (1.0f / inverseMass) * radius * radius;
    setInertiaTensor(Matrix3(I, 0, 0, 0, I, 0, 0, 0, I));
}

}
