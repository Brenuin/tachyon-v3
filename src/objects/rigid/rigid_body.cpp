#include "objects/rigid/rigid_body.h"

using namespace tachyon;

void RigidBody::setMass(real mass){
    if(mass > 0.0f){
        inverseMass = 1/mass;
    }
}
void RigidBody::setBoundingRadius(real r){
    boundingRadius = r;
}


void RigidBody::setInertiaTensor(const Matrix3& inertiaTensor){
    inverseInertia = inertiaTensor.inverse();
}

void RigidBody::clearAccumulators(){
    forceAccum.clear();
    torqueAccum.clear();
}

void RigidBody::addForce(const Vec3& force){
    forceAccum += force;
}

void RigidBody::addTorque(const Vec3& torque){
    torqueAccum += torque;
}

void RigidBody::addForceAtPoint(const Vec3& force, const Vec3& point){
    Vec3 r = point - pos;
    torqueAccum += r.cross(force);
    forceAccum += force;
}

Vec3 RigidBody::getVelocityAtPoint(const Vec3& localPoint) const{
    Vec3 r = localPoint - pos;
    return vel + angVel.cross(r);
}

void RigidBody::transformInertiaTensor(Matrix3& iworld, const Quaternion& q, const Matrix3& ibody, const Matrix4& rotMat) {
    Matrix3 R = q.toMatrix3();

    // iworld = R * ibody * R^T
    iworld = R * ibody * R.transpose();
}


void RigidBody::calculateDerivedData(){
    orient.normalize();
    transformInertiaTensor(inverseInertiaWorld, orient, inverseInertia, renderMatrix);
    renderMatrix = Matrix4::fromTransform(orient, pos);
}

// --- Classical Mechanics ---
// a = F / m                            → linear acceleration from accumulated force
// α = τ / I                           → angular acceleration from accumulated torque
// v += a * dt                         → linear velocity update
// ω += α * dt                         → angular velocity update
// x += v * dt                         → position update

// --- Orientation & Rotation ---
// q += 0.5 * q * ω * dt               → orientation update via quaternion integration
// inverseInertiaWorld = R * I^-1 * R^T → world-space inertia tensor update
// renderMatrix = [ R | x ]            → transformation matrix from orientation and position
void RigidBody::integrate(real duration){
    if(!isAwake || inverseMass == 0.0f) return;
    
    //linear
    prevAcc = acc;
    acc = forceAccum * inverseMass;
    vel += acc*duration;
    pos += vel*duration;

    //angular
    Matrix3 iworld = inverseInertiaWorld;
    Vec3 angAcc = iworld * torqueAccum;
    angVel += angAcc*duration;
    angVel *= real_pow(angularDamp,duration);
    orient.integrateAngularVelocity(angVel,duration);

    //stored pos/orient in matrix4 for rendering
    calculateDerivedData();
    clearAccumulators();
}

