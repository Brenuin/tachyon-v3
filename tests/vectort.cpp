#include <iostream>
#include <cassert>
#include <cmath>
#include "core/vec3.h"


using tachyon::Vec3;
using tachyon::real;

const real EPSILON = 1e-5;

bool almost_equal(real a, real b, real epsilon = EPSILON) {
    return std::fabs(a - b) < epsilon;
}

void test_addition() {
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{4.0f, 5.0f, 6.0f};
    Vec3 c = a + b;
    assert(almost_equal(c.x, 5.0f));
    assert(almost_equal(c.y, 7.0f));
    assert(almost_equal(c.z, 9.0f));
    std::cout << "Addition passed.\n";
}

void test_subtraction() {
    Vec3 a{5.0f, 7.0f, 9.0f};
    Vec3 b{1.0f, 2.0f, 3.0f};
    Vec3 c = a - b;
    assert(almost_equal(c.x, 4.0f));
    assert(almost_equal(c.y, 5.0f));
    assert(almost_equal(c.z, 6.0f));
    std::cout << "Subtraction passed.\n";
}

void test_scalar_multiplication() {
    Vec3 a{1.0f, -2.0f, 0.5f};
    Vec3 c = a * 2.0f;
    assert(almost_equal(c.x, 2.0f));
    assert(almost_equal(c.y, -4.0f));
    assert(almost_equal(c.z, 1.0f));
    std::cout << "Scalar multiplication passed.\n";
}

void test_dot_product() {
    Vec3 a{1.0f, 0.0f, 0.0f};
    Vec3 b{0.0f, 1.0f, 0.0f};
    real dot = a.dot(b);
    assert(almost_equal(dot, 0.0f));
    std::cout << "Dot product orthogonal passed.\n";

    Vec3 c{1.0f, 2.0f, 3.0f};
    Vec3 d{4.0f, -5.0f, 6.0f};
    real dot2 = c.dot(d);
    assert(almost_equal(dot2, 12.0f)); // 1*4 + 2*(-5) + 3*6
    std::cout << "Dot product general passed.\n";
}

void test_cross_product() {
    Vec3 a{1.0f, 0.0f, 0.0f};
    Vec3 b{0.0f, 1.0f, 0.0f};
    Vec3 c = a.cross(b);
    assert(almost_equal(c.x, 0.0f));
    assert(almost_equal(c.y, 0.0f));
    assert(almost_equal(c.z, 1.0f));
    std::cout << "Cross product passed.\n";
}

void test_magnitude_and_normalization() {
    Vec3 a{3.0f, 4.0f, 0.0f};
    real mag = a.magnitude();
    assert(almost_equal(mag, 5.0f));

    a.normalize();
    assert(almost_equal(a.magnitude(), 1.0f));
    std::cout << "Magnitude and normalization passed.\n";
}

void test_vector_negation() {
    Vec3 a{1.0f, -2.0f, 3.0f};
    Vec3 b = -a;
    assert(almost_equal(b.x, -1.0f));
    assert(almost_equal(b.y, 2.0f));
    assert(almost_equal(b.z, -3.0f));
    std::cout << "Negation passed.\n";
}

int main() {
    test_addition();
    test_subtraction();
    test_scalar_multiplication();
    test_dot_product();
    test_cross_product();
    test_magnitude_and_normalization();
    test_vector_negation();

    std::cout << "All vectort tests passed.\n";
    return 0;
}
