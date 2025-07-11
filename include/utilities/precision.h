#pragma once

#include <cstdint>
#include <cmath>
#include <limits>

//Set the precision of the engines numbers 0 = float 1.4E-45 to 3.4E+38 ;;  1 = double 4.9E-324 to 1.8E+308
#define TACHYON_USE_DOUBLE_PRECISION 0

namespace tachyon {

#if TACHYON_USE_DOUBLE_PRECISION
    typedef double real;
    #define REAL_MAX       DBL_MAX
    #define real_sqrt      sqrt
    #define real_abs       fabs
    #define real_sin       sin
    #define real_cos       cos
    #define real_exp       exp
    #define real_pow       pow
    #define real_fmod      fmod
    #define real_epsilon   DBL_EPSILON
    #define R_PI           3.14159265358979
#else
    typedef float real;
    #define REAL_MAX       FLT_MAX
    #define real_sqrt      sqrtf
    #define real_abs       fabsf
    #define real_sin       sinf
    #define real_cos       cosf
    #define real_exp       expf
    #define real_pow       powf
    #define real_fmod      fmodf
    #define real_epsilon   FLT_EPSILON
    #define R_PI           3.14159f
#endif

inline bool real_isfinite(real v) {
    return std::isfinite(v);
}

inline bool real_isinf(real v) {
    return std::isinf(v);
}

typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef real TimeStep;
typedef real Mass;
typedef real Length;
typedef u32  EntityID;
typedef u64  ConstraintHandle;

static constexpr real PI = static_cast<real>(3.14159265358979323846);
static constexpr real HALF_PI = PI * 0.5;
static constexpr real TAU = PI * 2;

static constexpr real SLEEP_EPSILON = static_cast<real>(0.01);
static constexpr real RESTITUTION_THRESHOLD = static_cast<real>(0.5);

inline bool is_valid(real v) {
    return real_isfinite(v) && !real_isinf(v);
}

}
