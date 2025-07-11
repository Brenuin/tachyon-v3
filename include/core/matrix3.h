#pragma once
#include "vec3.h"

namespace tachyon {

class Matrix3 {
public:
    real m[3][3];

    Matrix3() {
        setZero();
    }

    Matrix3(real m00, real m01, real m02,
            real m10, real m11, real m12,
            real m20, real m21, real m22) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22;
    }

    void setZero() {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                m[r][c] = 0.0;
    }
    static Matrix3 zero() {
        Matrix3 mat;
        mat.setZero();
        return mat;
    }

    void setIdentity() {
        setZero();
        m[0][0] = m[1][1] = m[2][2] = 1.0;
    }

    static Matrix3 identity() {
        Matrix3 mat;
        mat.setIdentity();
        return mat;
    }
    Vec3 getColumn(int i) const {
    return Vec3(m[0][i], m[1][i], m[2][i]);
    }

    Matrix3 transpose() const {
        return Matrix3(
            m[0][0], m[1][0], m[2][0],
            m[0][1], m[1][1], m[2][1],
            m[0][2], m[1][2], m[2][2]
        );
    }

    real determinant() const {
        return
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }

    Matrix3 inverse() const {
        real det = determinant();
        if (std::fabs(det) < 1e-6) return Matrix3();

        real invDet = 1.0 / det;

        Matrix3 inv;
        inv.m[0][0] =  (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * invDet;
        inv.m[0][1] = -(m[0][1]*m[2][2] - m[0][2]*m[2][1]) * invDet;
        inv.m[0][2] =  (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * invDet;

        inv.m[1][0] = -(m[1][0]*m[2][2] - m[1][2]*m[2][0]) * invDet;
        inv.m[1][1] =  (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * invDet;
        inv.m[1][2] = -(m[0][0]*m[1][2] - m[0][2]*m[1][0]) * invDet;

        inv.m[2][0] =  (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * invDet;
        inv.m[2][1] = -(m[0][0]*m[2][1] - m[0][1]*m[2][0]) * invDet;
        inv.m[2][2] =  (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * invDet;

        return inv;
    }


    Vec3 operator*(const Vec3& v) const {
        return Vec3(
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z
        );
    }


    Matrix3 operator*(const Matrix3& other) const {
        Matrix3 result;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                result.m[r][c] = 
                    m[r][0]*other.m[0][c] +
                    m[r][1]*other.m[1][c] +
                    m[r][2]*other.m[2][c];
        return result;
    }
     Matrix3 operator*(real scalar) const {
        Matrix3 result;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                result.m[r][c] = m[r][c] * scalar;
        return result;
    }
};
   

} 
