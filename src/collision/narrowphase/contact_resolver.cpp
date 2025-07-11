#pragma once
#include <iostream>
#include "collision/narrowphase/contact_resolver.h"

namespace tachyon{
void ContactResolver::resolveContactsSimple(std::vector<Contact>& contacts, real duration){
    for(auto& contact: contacts){
        resolveInterpenetration(contact); // Baumgarte replaces this
        resolveVelocity(contact, duration);
    }
}

void ContactResolver::resolveContactsIterative(std::vector<Contact>& contacts, real duration, unsigned maxIterations){
    const unsigned MAX_PER_CONTACT = 5;
    unsigned iterationsUsed = 0;
    while(iterationsUsed < maxIterations){
        real maxPenetration = 0.0f;
        int contactIndex = -1;
        for(size_t i = 0; i < contacts.size(); i++){
            if(contacts[i].resolveCount >= MAX_PER_CONTACT) continue;
            if(contacts[i].penetration > maxPenetration){
                maxPenetration = contacts[i].penetration;
                contactIndex = static_cast<int>(i);
            }
        }

        if(contactIndex == -1) break;
        resolveVelocity(contacts[contactIndex], duration);
        
        if (contacts[contactIndex].bodyA->angVel.magnitude() > 0.01f || contacts[contactIndex].bodyB->angVel.magnitude() > 0.01f) {

        }
        resolveInterpenetration(contacts[contactIndex]);
        contacts[contactIndex].resolveCount++;
        ++iterationsUsed;
    }
}

// 
// j = -(1 + e) * (v_rel · n + bias) / (Σ_invMass + rotational terms)
// impulse = j * n
// 
// Baumgarte stabilization: bias = β * max(penetration - slop, 0) / dt
void ContactResolver::resolveVelocity(Contact& contact, real duration){
    RigidBody* a = contact.bodyA;
    RigidBody* b = contact.bodyB;
    Vec3 ra = contact.contactPoint - a->pos;
    Vec3 rb = contact.contactPoint - b->pos;
    Vec3 va = a->vel + a->angVel.cross(ra);
    Vec3 vb = b->vel + b->angVel.cross(rb);
    Vec3 v_rel = vb - va;

    real velAlongNormal = v_rel.dot(contact.contactNormal);
    if (velAlongNormal > 0.0f) return;

    const real VELOCITY_EPSILON = 0.01f;
    if (velAlongNormal > -VELOCITY_EPSILON) return;

    Vec3 raCrossN = ra.cross(contact.contactNormal);
    Vec3 rbCrossN = rb.cross(contact.contactNormal);
    real invMassSum = a->inverseMass + b->inverseMass +
        contact.contactNormal.dot((a->inverseInertiaWorld * raCrossN).cross(ra)) +
        contact.contactNormal.dot((b->inverseInertiaWorld * rbCrossN).cross(rb));

    if (invMassSum <= 0.0f) return;

    //Baumgarte bias stabilization
    const real beta = 0.2f;
    const real slop = 0.01f;
    real baumgarteBias = beta * std::max(contact.penetration - slop, 0.0f) / duration;

    real j = -(velAlongNormal + baumgarteBias) * (1 + contact.restitution) / invMassSum;
    Vec3 impulse = contact.contactNormal * j;

    a->vel -= impulse * a->inverseMass;
    b->vel += impulse * b->inverseMass;

    a->angVel -= a->inverseInertiaWorld * ra.cross(impulse);
    b->angVel += b->inverseInertiaWorld * rb.cross(impulse);
}


// x_a -= (penetration * m_b / (m_a + m_b)) * n
// x_b += (penetration * m_a / (m_a + m_b)) * n
void ContactResolver::resolveInterpenetration(Contact& contact){
    const real beta = 0.15f;
        
    RigidBody* a = contact.bodyA;
    RigidBody* b = contact.bodyB;
    real totalInverseMass = a->inverseMass + b->inverseMass;
    if (totalInverseMass <= 0.0f) return;
    Vec3 movePerInvMass = contact.contactNormal * (contact.penetration / totalInverseMass);
    a->pos -= beta * movePerInvMass * a->inverseMass;
    b->pos += beta * movePerInvMass * b->inverseMass;
    
     Vec3 separation = b->pos - a->pos;
    contact.penetration *= 0.85;
}
}