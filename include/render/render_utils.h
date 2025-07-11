#pragma once
#include "core/vec3.h"
#include "core/matrix4.h"
#include "utilities/precision.h"

namespace tachyon {

// Renders a 2D circle at position pos with given radius (use for flat-sphere tests)
void drawCircle2D(const Vec3& pos, real radius);

// (Optional) 3D sphere rendering using OpenGL with model transform
void drawSphere3D(const Vec3& pos, real radius);

void drawBox2D(const Matrix4& transform, const Vec3& halfSize);

void drawBox3D(const Matrix4& transform, const Vec3& halfSize);

void drawGrid3D(float spacing = 1.0f, int count = 20);

}
