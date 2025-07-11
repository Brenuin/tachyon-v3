#include "objects/particles/particle.h"


using namespace tachyon;


particle::particle()
    : position(), velocity(), acceleration(), forceAccum(),
      inverseMass(1.0f), damping(0.99f), radius(1.0f),
      previousPosition(), initialized(false) {}

            void particle::integrate(real dt) {
                if (inverseMass <= 0.0f) return;
                position += velocity * dt;
                Vec3 resultingAcc = acceleration;
                resultingAcc += forceAccum * inverseMass;
                velocity += resultingAcc * dt;
                velocity *= real_pow(damping, dt);
                clearAccumulator();
            }
    
            void particle::clearAccumulator(){
                forceAccum = Vec3();
            }
            void particle::addForce(const Vec3 force){
                forceAccum += force;
            }

            void particle::setPosition(const Vec3& p){
                position = p;
            }
            void particle::setPosition(real x, real y, real z){
                position = Vec3(x,y,z);
            }
            Vec3 particle::getPosition() const{
                return position;
            }

            void particle::setVelocity(const Vec3& v){
                velocity = v;
            }
            void particle::setVelocity(real x, real y, real z){
                velocity = Vec3(x,y,z);
            }
            Vec3 particle::getVelocity() const{
                return velocity;
            }
            void particle::setAcceleration(const Vec3& a){
                acceleration = a;
            }
            void particle::setAcceleration(real x, real y, real z){
                acceleration = Vec3(x,y,z);
            }
            Vec3 particle::getAcceleration() const{
                return acceleration;
            }

            void particle::setDamping(real d){
                damping = d;
            }
            real particle::getDamping() const{
                return damping;
            }

            void particle::setMass(real m){
                inverseMass = 1/m;
            }
            real particle::getMass() const{
                if (inverseMass == 0) return INFINITY;
                return 1.0f / inverseMass;
            }
            real particle::getInverseMass() const{
                    return inverseMass;
            }
            bool particle::hasFiniteMass() const{
                return inverseMass > 0.0f;
            }
            void particle::updateGPUData(){
                gpuData.invMass = static_cast<float>(inverseMass);
                gpuData.damp = static_cast<float>(damping);
                gpuData.acc = make_float3(
                    static_cast<float>(acceleration.x),
                    static_cast<float>(acceleration.y),
                    static_cast<float>(acceleration.z)
                );
                gpuData.vel = make_float3(
                    static_cast<float>(velocity.x),
                    static_cast<float>(velocity.y),
                    static_cast<float>(velocity.z)
                );
                gpuData.pos = make_float3(
                    static_cast<float>(position.x),
                    static_cast<float>(position.y),
                    static_cast<float>(position.z)
                );
                gpuData.fac = make_float3(
                    static_cast<float>(forceAccum.x),
                    static_cast<float>(forceAccum.y),
                    static_cast<float>(forceAccum.z)
                );
                
            }
