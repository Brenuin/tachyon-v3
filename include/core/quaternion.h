#pragma once
#include "vec3.h"
#include "matrix3.h"

namespace tachyon {

class Quaternion {
public:
    real w, x, y, z;

    Quaternion() : w(1), x(0), y(0), z(0) {}

    Quaternion(real w, real x, real y, real z)
        : w(w), x(x), y(y), z(z) {}

    Quaternion(real angleRad, const Vec3& axis) {
        Vec3 normAxis = axis.normalized();
        real halfAngle = angleRad * 0.5;
        real s = real_sin(halfAngle);
        w = real_cos(halfAngle);
        x = normAxis.x * s;
        y = normAxis.y * s;
        z = normAxis.z * s;
    }

    void normalize() {
        real mag = real_sqrt(w*w + x*x + y*y + z*z);
        if (mag > 0.0f) {
            w /= mag;
            x /= mag;
            y /= mag;
            z /= mag;
        }
    }

    Quaternion conjugate() const {
        return Quaternion(w, -x, -y, -z);
    }

    Quaternion inverse() const {
    return conjugate(); 
    }
    
    Vec3 rotate(const Vec3& v) const {
    Quaternion qv(0, v.x, v.y, v.z);
    Quaternion result = (*this) * qv * this->conjugate();
    return Vec3(result.x, result.y, result.z);
    }

    Quaternion operator*(const Quaternion& q) const {
        return Quaternion(
            w*q.w - x*q.x - y*q.y - z*q.z,
            w*q.x + x*q.w + y*q.z - z*q.y,
            w*q.y - x*q.z + y*q.w + z*q.x,
            w*q.z + x*q.y - y*q.x + z*q.w
        );
    }

    Quaternion operator*(real scale) const {
        return Quaternion(w * scale, x * scale, y * scale, z * scale);
    }

    Quaternion& operator+=(const Quaternion& dq) {
        w += dq.w;
        x += dq.x;
        y += dq.y;
        z += dq.z;
        return *this;
    }

    // Integrate angular velocity into orientation
    void integrateAngularVelocity(const Vec3& omega, real dt) {
        Quaternion omegaQ(0, omega.x, omega.y, omega.z);
        Quaternion dq = (*this * omegaQ) * (0.5f * dt);
        *this += dq;
        normalize();
    }

    static Quaternion identity() {
        return Quaternion(1, 0, 0, 0);
    }

    // Convert to 3x3 rotation matrix
    Matrix3 toMatrix3() const {
        real tx = 2.0f * x;
        real ty = 2.0f * y;
        real tz = 2.0f * z;
        real twx = tx * w;
        real twy = ty * w;
        real twz = tz * w;
        real txx = tx * x;
        real txy = ty * x;
        real txz = tz * x;
        real tyy = ty * y;
        real tyz = tz * y;
        real tzz = tz * z;

        return Matrix3(
            1 - (tyy + tzz), txy - twz,      txz + twy,
            txy + twz,      1 - (txx + tzz), tyz - twx,
            txz - twy,      tyz + twx,      1 - (txx + tyy)
        );
    }
};

} // namespace tachyon
