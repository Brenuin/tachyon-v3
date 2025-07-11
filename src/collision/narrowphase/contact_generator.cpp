#include "collision/narrowphase/contact_generator.h"
#include <cmath>
#include <iostream>
#include <algorithm> 
namespace tachyon {

std::vector<Contact> ContactGenerator::generateContacts(const std::vector<std::pair<RigidBody*, RigidBody*>>& pairs) {
    std::vector<Contact> contacts;

    for (const auto& [a, b] : pairs) {
        Contact contact;

        ShapeType typeA = a->getShapeType();
        ShapeType typeB = b->getShapeType();

        if (typeA == ShapeType::Sphere && typeB == ShapeType::Sphere) {
            if (detectSphereSphere(a, b, contact))
                contacts.push_back(contact);
        }
        else if (typeA == ShapeType::Sphere && typeB == ShapeType::Box) {
            if (detectSphereBox(a, b, contact))
                contacts.push_back(contact);
        }
        else if (typeA == ShapeType::Box && typeB == ShapeType::Sphere) {
            if (detectSphereBox(b, a, contact)) {
                std::swap(contact.bodyA, contact.bodyB);
                contact.contactNormal = -contact.contactNormal;
                contacts.push_back(contact);
            }
        }
        else if (typeA == ShapeType::Box && typeB == ShapeType::Box) {
            if (detectBoxBox(a, b, contact))
                contacts.push_back(contact);
        }
    }

    return contacts;
}

bool ContactGenerator::detectSphereSphere(RigidBody* a, RigidBody* b, Contact& out) {
    Vec3 dist = b->pos - a->pos;
    real distMag = dist.magnitude();
    real rSum = a->boundingRadius + b->boundingRadius;

    if (distMag >= rSum) return false;

    Vec3 normal = (distMag > 0.0f) ? dist / distMag : Vec3(1.0f, 0.0f, 0.0f);

    out.bodyA = a;
    out.bodyB = b;
    out.contactNormal = normal;
    out.penetration = rSum - distMag;
    out.contactPoint = a->pos + normal * (a->boundingRadius - 0.5f * out.penetration);
    out.restitution = 0.8f;
    out.friction = 0.0f;

    return true;
}

bool ContactGenerator::detectSphereBox(RigidBody* sphere, RigidBody* box, Contact& out) {
    Box* b = static_cast<Box*>(box);
    Vec3 localCenter = box->toLocal(sphere->pos);  // Step 1: move sphere center into box space

    const Vec3& half = b->getHalfExtents();
    real radius = sphere->boundingRadius;


    // Step 2: Early-out Separating Axis Test
    if (std::abs(localCenter.x) - radius > half.x ||
        std::abs(localCenter.y) - radius > half.y ||
        std::abs(localCenter.z) - radius > half.z) {

        return false;
    }

    // Step 3: Clamp local center to box extents (find closest point)
    Vec3 closest = localCenter;
    closest.x = std::clamp(closest.x, -half.x, half.x);
    closest.y = std::clamp(closest.y, -half.y, half.y);
    closest.z = std::clamp(closest.z, -half.z, half.z);

    Vec3 delta = closest - localCenter;
    real distSq = delta.magnitudeSquared();



    if (distSq > radius * radius) {
        return false;
    }

    // Step 4: Compute contact data in world space
    Vec3 closestWorld = box->toWorld(closest);
    Vec3 normal = sphere->pos - closestWorld;

    if (normal.magnitudeSquared() < 1e-8f || !std::isfinite(normal.x) || !std::isfinite(normal.y) || !std::isfinite(normal.z)) {
        normal = Vec3(1.0f, 0.0f, 0.0f);
    } else {
        normal.normalize();
    }

    real dist = std::sqrt(distSq);
    real penetration = std::max(0.0f, radius - dist);

    if (!std::isfinite(penetration) || penetration > 10.0f) {
        return false;
    }
   

    // Relative velocity check
    Vec3 relVel = sphere->vel - box->vel;
    real sepVel = relVel.dot(normal);


    // Optional: Ensure normal faces from box to sphere
    if (normal.dot(sphere->pos - box->pos) < 0) {
        
        normal = -normal;
    }

    out.bodyA = box;
    out.bodyB = sphere;
    out.contactPoint = closestWorld;
    out.contactNormal = normal;
    out.penetration = penetration;
    out.restitution = 0.9f;
    out.friction = 0.0f;

    return true;
}






bool ContactGenerator::detectBoxBox(RigidBody* a, RigidBody* b, Contact& out) {
    Box* boxA = static_cast<Box*>(a);
    Box* boxB = static_cast<Box*>(b);

    Matrix3 rotA = a->orient.toMatrix3();
    Matrix3 rotB = b->orient.toMatrix3();
    Vec3 heA = boxA->getHalfExtents();
    Vec3 heB = boxB->getHalfExtents();
    Vec3 posA = a->pos;
    Vec3 posB = b->pos;

    Matrix3 R;
    real EPSILON = 1e-6;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R.m[i][j] = rotA.getColumn(i).dot(rotB.getColumn(j));

    Vec3 t = rotA.transpose() * (posB - posA);
    Matrix3 absR;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            absR.m[i][j] = std::abs(R.m[i][j]) + EPSILON;

    for (int i = 0; i < 3; i++) {
        real ra = heA[i];
        real rb = heB.x * absR.m[i][0] + heB.y * absR.m[i][1] + heB.z * absR.m[i][2];
        if (std::abs(t[i]) > ra + rb) return false;
    }
    for (int i = 0; i < 3; i++) {
        real ra = heA.x * absR.m[0][i] + heA.y * absR.m[1][i] + heA.z * absR.m[2][i];
        real rb = heB[i];
        if (std::abs(t.x * R.m[0][i] + t.y * R.m[1][i] + t.z * R.m[2][i]) > ra + rb) return false;
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            real ra = heA[(i + 1) % 3] * absR.m[(i + 2) % 3][j] +
                      heA[(i + 2) % 3] * absR.m[(i + 1) % 3][j];
            real rb = heB[(j + 1) % 3] * absR.m[i][(j + 2) % 3] +
                      heB[(j + 2) % 3] * absR.m[i][(j + 1) % 3];
            real val = std::abs(
                t[(i + 2) % 3] * R.m[(i + 1) % 3][j] -
                t[(i + 1) % 3] * R.m[(i + 2) % 3][j]
            );
            if (val > ra + rb) return false;
        }
    }

    out.bodyA = a;
    out.bodyB = b;
    if (!computeBoxBoxContactPoint(boxA, boxB, out)) return false;
    out.contactNormal = (posB - posA).normalized();
    out.penetration = 0.01f;  // Test with a large enough value
    out.restitution = 0.8f;
    out.friction = 0.3f;
    return true;
}


bool ContactGenerator::computeBoxBoxContactPoint(const Box* boxA, const Box* boxB, Contact& contact) {
    const RigidBody* a = boxA;
    const RigidBody* b = boxB;

    Matrix3 rotA = a->orient.toMatrix3();
    Matrix3 rotB = b->orient.toMatrix3();

    Vec3 heA = boxA->getHalfExtents();
    Vec3 heB = boxB->getHalfExtents();

    Vec3 posA = a->pos;
    Vec3 posB = b->pos;

    Vec3 dir = (posB - posA).normalized();

    Vec3 contactA = posA + rotA * Vec3(
        dir.dot(rotA.getColumn(0)) > 0 ? heA.x : -heA.x,
        dir.dot(rotA.getColumn(1)) > 0 ? heA.y : -heA.y,
        dir.dot(rotA.getColumn(2)) > 0 ? heA.z : -heA.z
    );

    Vec3 contactB = posB - rotB * Vec3(
        dir.dot(rotB.getColumn(0)) > 0 ? heB.x : -heB.x,
        dir.dot(rotB.getColumn(1)) > 0 ? heB.y : -heB.y,
        dir.dot(rotB.getColumn(2)) > 0 ? heB.z : -heB.z
    );

    contact.contactPoint = (contactA + contactB) * 0.5f;

    return true;
}

} 
