#include "render/render_utils.h"
#include <GLFW/glfw3.h>
#include <cmath>

namespace tachyon {

void drawCircle2D(const Vec3& pos, real radius) {
    const int segments = 32;
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < segments; ++i) {
        float theta = 2.0f * 3.1415926f * float(i) / float(segments);
        float dx = radius * std::cos(theta);
        float dy = radius * std::sin(theta);
        glVertex2f(pos.x + dx, pos.y + dy);
    }
    glEnd();
}

void drawSphere3D(const Vec3& pos, real radius) {
    const int stacks = 12;
    const int slices = 24;

    for (int i = 0; i < stacks; ++i) {
        float lat0 = PI * (-0.5f + (float)i / stacks);
        float z0 = std::sin(lat0);
        float zr0 = std::cos(lat0);

        float lat1 = PI * (-0.5f + (float)(i + 1) / stacks);
        float z1 = std::sin(lat1);
        float zr1 = std::cos(lat1);

        glBegin(GL_LINE_LOOP);
        for (int j = 0; j <= slices; ++j) {
            float lng = 2.0f * PI * (float)j / slices;
            float x = std::cos(lng);
            float y = std::sin(lng);

            glVertex3f(pos.x + radius * x * zr0,
                       pos.y + radius * y * zr0,
                       pos.z + radius * z0);

            glVertex3f(pos.x + radius * x * zr1,
                       pos.y + radius * y * zr1,
                       pos.z + radius * z1);
        }
        glEnd();
    }
}

void drawBox2D(const Matrix4& transform, const Vec3& halfSize) {
    Vec3 corners[4] = {
        Vec3(-halfSize.x, -halfSize.y, 0),
        Vec3( halfSize.x, -halfSize.y, 0),
        Vec3( halfSize.x,  halfSize.y, 0),
        Vec3(-halfSize.x,  halfSize.y, 0)
    };

    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 4; ++i) {
        Vec3 world = transform * corners[i];
        glVertex2f(world.x, world.y);
    }
    glEnd();
}

void drawBox3D(const Matrix4& transform, const Vec3& halfSize) {
    Vec3 corners[8] = {
        Vec3(-halfSize.x, -halfSize.y, -halfSize.z),
        Vec3( halfSize.x, -halfSize.y, -halfSize.z),
        Vec3( halfSize.x,  halfSize.y, -halfSize.z),
        Vec3(-halfSize.x,  halfSize.y, -halfSize.z),
        Vec3(-halfSize.x, -halfSize.y,  halfSize.z),
        Vec3( halfSize.x, -halfSize.y,  halfSize.z),
        Vec3( halfSize.x,  halfSize.y,  halfSize.z),
        Vec3(-halfSize.x,  halfSize.y,  halfSize.z)
    };

    static const int edges[12][2] = {
        {0,1}, {1,2}, {2,3}, {3,0}, // bottom face
        {4,5}, {5,6}, {6,7}, {7,4}, // top face
        {0,4}, {1,5}, {2,6}, {3,7}  // vertical edges
    };

    glBegin(GL_LINES);
    for (int i = 0; i < 12; ++i) {
        Vec3 p1 = transform * corners[edges[i][0]];
        Vec3 p2 = transform * corners[edges[i][1]];
        glVertex3f(p1.x, p1.y, p1.z);
        glVertex3f(p2.x, p2.y, p2.z);
    }
    glEnd();
}

void drawGrid3D(float spacing, int count ) {
    float extent = spacing * count * 0.5f;

    glColor3f(0.3f, 0.3f, 0.3f);
    glBegin(GL_LINES);

    for (int i = -count / 2; i <= count / 2; ++i) {
        // Lines along X
        glVertex3f((float)i * spacing, 0.0f, -extent);
        glVertex3f((float)i * spacing, 0.0f,  extent);

        // Lines along Z
        glVertex3f(-extent, 0.0f, (float)i * spacing);
        glVertex3f( extent, 0.0f, (float)i * spacing);
    }

    glEnd();
    glColor3f(1.0f, 1.0f, 1.0f); // Reset color
}


}
