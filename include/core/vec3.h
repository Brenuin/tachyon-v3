#pragma once
#include <ostream>
#include "utilities/precision.h"


namespace tachyon {

    // A 3D vector with real-valued components.
    // Includes vector arithmetic, dot/cross products, normalization, and scalar ops.

    class Vec3 {
    public:
        real x, y, z;

    private:
        real pad;

    public:
        Vec3() : x(0), y(0), z(0) {}
        Vec3(const real x, const real y, const real z) : x(x), y(y), z(z) {}
        Vec3(real value) : x(value), y(value), z(value) {}


        void invert() {
            x = -x;
            y = -y;
            z = -z;
        }

        real magnitude() const {
            return real_sqrt(x * x + y * y + z * z);
        }

        real magnitudeSquared() const {
            return x * x + y * y + z * z;
        }

        void normalize() {
            real mag = magnitude();
            if (mag > 0) {
                x /= mag;
                y /= mag;
                z /= mag;
            }
        }

        Vec3 normalized() const {
            Vec3 copy = *this;
            copy.normalize();
            return copy;
        }

        void clear() {
            x = y = z = 0.0f;
        }
        //component product (uselss)
        Vec3 componentProduct(const Vec3 b) const{
            return Vec3(x*b.x, y*b.y, z*b.z);
        }

        // Scalar multiplication (in-place)
        Vec3& operator*=(const real v) {
            x *= v;
            y *= v;
            z *= v;
            return *this;
        }

        // Scalar multiplication (returns new)
        Vec3 operator*(const real v) const {
            return Vec3(x * v, y * v, z * v);
        }

        // Dot product of two vectors:
        // a · b = a.x * b.x + a.y * b.y + a.z * b.z
        //       = |a||b|cos(θ)
        real dot(const Vec3& b) const {
            return x * b.x + y * b.y + z * b.z;
        }

        // Optional: operator* for dot (not recommended long term)
        real operator*(const Vec3& b) const {
            return x * b.x + y * b.y + z * b.z;
        }

        // Vector addition
        Vec3 operator+(const Vec3& b) const {
            return Vec3(x + b.x, y + b.y, z + b.z);
        }

        Vec3& operator+=(const Vec3& b) {
            x += b.x;
            y += b.y;
            z += b.z;
            return *this;
        }

        // Vector subtraction
        Vec3 operator-(const Vec3& b) const {
            return Vec3(x - b.x, y - b.y, z - b.z);
        }

        Vec3& operator-=(const Vec3& b) {
            x -= b.x;
            y -= b.y;
            z -= b.z;
            return *this;
        }

        // Unary minus (returns negated copy)
        Vec3 operator-() const {
            return Vec3(-x, -y, -z);
        }

        // Cross product of two vectors:
        //           |  i   j   k  |
        //   a × b = | a.x a.y a.z |
        //           | b.x b.y b.z |
        //
        // = (a.y * b.z - a.z * b.y) * i
        // + (a.z * b.x - a.x * b.z) * j
        // + (a.x * b.y - a.y * b.x) * k
        Vec3 cross(const Vec3& b) const {
            return Vec3(
                y * b.z - z * b.y,
                z * b.x - x * b.z,
                x * b.y - y * b.x
            );
        }
        Vec3 operator% (const Vec3& b) const {
            return Vec3(
                y * b.z - z * b.y,
                z * b.x - x * b.z,
                x * b.y - y * b.x
            );
        }

        Vec3& operator%=(const Vec3& b) {
            *this = cross(b);
            return *this;
        }
        //Divison 
        Vec3 operator/(const real v)const{
            return Vec3(x/v,y/v,z/v);
        }
        Vec3 operator/=(const real v) {
            x /= v;
            y /= v;
            z /= v;
            return *this;
        }
        real& operator[](int i) {
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }

        const real& operator[](int i) const {
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }

    };

    // Left-hand scalar multiplication
    inline Vec3 operator*(real v, const Vec3& b) {
        return Vec3(b.x * v, b.y * v, b.z * v);
    }
    inline std::ostream& operator<<(std::ostream& os, const Vec3& v) {
        os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return os;
    }


}
