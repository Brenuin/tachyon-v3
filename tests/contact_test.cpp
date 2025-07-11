#include <iostream>
#include "objects/rigid/sphere.h"
#include "collision/narrowphase/contact_generator.h"
#include "collision/narrowphase/contact_resolver.h"

using namespace tachyon;

void printVec3(const char* label, const Vec3& v) {
    std::cout << label << ": (" << v.x << ", " << v.y << ", " << v.z << ")\n";
}

int main() {
    // Create two spheres
    Sphere bodyA(1.0f, 1.0f); // radius, mass
    Sphere bodyB(1.0f, 1.0f);

    // Slightly overlap them
    bodyA.pos = Vec3(-0.9f, 0.0f, 0.0f);
    bodyB.pos = Vec3(0.9f, 0.0f, 0.0f);

    // Send them at each other
    bodyA.vel = Vec3(2.0f, 0.0f, 0.0f);
    bodyB.vel = Vec3(-2.0f, 0.0f, 0.0f);

    bodyA.calculateDerivedData();
    bodyB.calculateDerivedData();

    // Generate contact
    ContactGenerator generator;
    std::vector<std::pair<RigidBody*, RigidBody*>> pairs = { { &bodyA, &bodyB } };
    std::vector<Contact> contacts = generator.generateContacts(pairs);

    if (contacts.empty()) {
        std::cout << "No contacts generated.\n";
        return 1;
    }

    ContactResolver resolver;

    std::cout << "=== Before Resolution ===\n";
    printVec3("BodyA Pos", bodyA.pos);
    printVec3("BodyA Vel", bodyA.vel);
    printVec3("BodyB Pos", bodyB.pos);
    printVec3("BodyB Vel", bodyB.vel);

    resolver.resolveContactsSimple(contacts, 1.0f);

    std::cout << "\n=== After Resolution ===\n";
    printVec3("BodyA Pos", bodyA.pos);
    printVec3("BodyA Vel", bodyA.vel);
    printVec3("BodyB Pos", bodyB.pos);
    printVec3("BodyB Vel", bodyB.vel);

    return 0;
}
