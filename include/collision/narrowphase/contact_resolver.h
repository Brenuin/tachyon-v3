#pragma once
#include "contact.h"
#include <vector>

namespace tachyon{

class ContactResolver{
    
public:
    void resolveContactsSimple(std::vector<Contact>& contacts, real duration);
    void resolveContactsIterative(std::vector<Contact>& contacts, real duration, unsigned maxIterations);


private:
    void resolveVelocity(Contact& contact, real duration);
    void resolveInterpenetration(Contact& contact);
};

    
}