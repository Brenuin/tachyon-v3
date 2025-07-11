#pragma once
#include "vec3.h"
#include "matrix3.h"
#include "quaternion.h"

namespace tachyon {

class Matrix4 {
public:
    real m[4][4];

    Matrix4() {
        setIdentity();
    }

    void setZero() {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                m[r][c] = 0.0;
    }

    void setIdentity() {
        setZero();
        m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0;
    }

    void setRotation(const Matrix3& rot) {
    m[0][0] = rot.m[0][0]; m[0][1] = rot.m[0][1]; m[0][2] = rot.m[0][2]; m[0][3] = 0.0f;
    m[1][0] = rot.m[1][0]; m[1][1] = rot.m[1][1]; m[1][2] = rot.m[1][2]; m[1][3] = 0.0f;
    m[2][0] = rot.m[2][0]; m[2][1] = rot.m[2][1]; m[2][2] = rot.m[2][2]; m[2][3] = 0.0f;
    m[3][0] = 0.0f;        m[3][1] = 0.0f;        m[3][2] = 0.0f;        m[3][3] = 1.0f;
}

    void setTranslation(const Vec3& pos) {
        m[0][3] = pos.x;
        m[1][3] = pos.y;
        m[2][3] = pos.z;
    }


    static Matrix4 fromTransform(const Quaternion& q, const Vec3& pos) {
        Matrix4 result;
        result.setRotation(q.toMatrix3());
        result.setTranslation(pos);
        result.m[3][0] = result.m[3][1] = result.m[3][2] = 0.0;
        result.m[3][3] = 1.0;
        return result;
    }

  
    void toOpenGLArray(real* out) const {
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
                out[c * 4 + r] = m[r][c];
    }
    
    Vec3 transformDirection(const Vec3& v) const {
    return Vec3(
        m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
        m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
        m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z
    );
}

    Vec3 operator*(const Vec3& v) const {
    return Vec3(
        m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3],  // x row
        m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3],  // y row
        m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3]   // z row
    );
}

};

} 