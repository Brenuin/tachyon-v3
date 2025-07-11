#pragma once
#include "core/vec3.h"
#include "objects/rigid/rigid_body.h"
#include "objects/rigid/box.h"
#include "objects/rigid/sphere.h"
#include "contact.h"
#include <vector>
#include <utility>


namespace tachyon{

class ContactGenerator{
public:
    std::vector<Contact> generateContacts(const std::vector<std::pair<RigidBody*, RigidBody*>>& pairs);

private:
    bool detectSphereSphere(RigidBody* a, RigidBody* b, Contact& out);
    bool detectSphereBox(RigidBody* sphere, RigidBody* box, Contact& out);
    bool detectBoxBox(RigidBody* a, RigidBody* b, Contact& out);
    bool computeBoxBoxContactPoint(const Box* boxA, const Box* boxB, Contact& contact);




};
}