#pragma once
#include "core/vec3.h"
#include "core/matrix3.h"
#include "core/matrix4.h"
#include "core/quaternion.h" 

namespace tachyon {

    enum class ShapeType { Sphere, Box, Plane };

class RigidBody {
public:
    // Physical state
    Vec3 pos;              // Position (center of mass)
    Vec3 vel;              // Linear velocity
    Vec3 acc;              // Linear acceleration
    Vec3 angVel;           // Angular velocity
    Vec3 forceAccum;       // Accumulated external forces
    Vec3 torqueAccum;      // Accumulated torques
    Vec3 prevAcc;          // Acceleration from last frame (optional use)

    // Orientation
    Quaternion orient;     // Orientation as a quaternion

    // Inertia
    Matrix3 inverseInertia;       // Body-space inverse inertia tensor
    Matrix3 inverseInertiaWorld;  // World-space version

    // Rendering transform
    Matrix4 renderMatrix;

    // Mass/damping
    real inverseMass;
    real linearDamp;
    real angularDamp;

    //BroadPhase
    real boundingRadius;

    // State
    bool isAwake = true;

    //Shape type      
    virtual ShapeType getShapeType() const = 0;

    // Core behavior
    void calculateDerivedData();
    void integrate(real duration);
    void setMass(real mass);
    void setBoundingRadius(real r);
    void setInertiaTensor(const Matrix3& inertiaTensor);

    void addForce(const Vec3& force);
    void addTorque(const Vec3& torque);
    void addForceAtPoint(const Vec3& force, const Vec3& point);
    void clearAccumulators();

    Vec3 getVelocityAtPoint(const Vec3& localPoint) const;

    //Transforms 
    Vec3 toLocal(const Vec3& worldPoint) const {
        return orient.inverse().rotate(worldPoint - pos);
    }

    Vec3 toWorld(const Vec3& localPoint) const {
        return pos + orient.rotate(localPoint);
    }

private:
    void transformInertiaTensor(Matrix3& iworld, const Quaternion& q, const Matrix3& ibody, const Matrix4& rotMat);
};

}
