#include "render/render_vec3.h"
#include "core/vec3.h"
#include <vector>
#include <cmath>

using tachyon::Vec3;
using tachyon::RenderVec3;

int main() {
    RenderVec3 renderer(800, 800, "Vec3 Viewer");
    Vec3 origin = {0.0f, 0.0f, 0.0f};

    std::vector<std::pair<Vec3, Vec3>> vectors = {
        {origin, {2.0f, 3.0f, 1.0f}},     // upward right
        {origin, {-4.0f, 1.0f, -2.0f}},   // backward left
        {origin, {1.0f, -2.0f, 3.0f}},    // deep diagonal
        {origin, {-3.0f, -3.0f, 0.0f}},   // down-left
        {origin, {0.5f, 5.0f, -1.0f}},    // steep
        {origin, {-2.5f, -1.5f, 3.5f}},   // messy upward
        {origin, {4.0f, -2.0f, 2.0f}},    // wide arc
        {origin, {-1.0f, 0.0f, 6.0f}}     // z spike
    };

    std::vector<Vec3> colors = {
        {1.0f, 1.0f, 0.0f}, // yellow
        {0.0f, 1.0f, 1.0f}, // cyan
        {1.0f, 0.0f, 1.0f}, // magenta
        {1.0f, 0.5f, 0.0f}, // orange
        {0.0f, 0.5f, 1.0f}, // sky blue
        {0.5f, 1.0f, 0.0f}, // lime
        {1.0f, 0.0f, 0.0f}, // red
        {0.5f, 0.0f, 1.0f}  // purple
    };

    // Generate a circular spread of vectors
    for (int i = 0; i < 12; ++i) {
        float angle = i * 3.14159f / 6.0f;
        float x = std::cos(angle) * 3.0f;
        float y = std::sin(angle) * 3.0f;
        vectors.push_back({origin, {x, y, 0.0f}});
        colors.push_back({0.2f + 0.6f * std::abs(x) / 3.0f, 0.2f, 1.0f});
    }

    while (!renderer.shouldClose()) {
        renderer.beginFrame();
        renderer.drawAxis(10.0f);

        for (size_t i = 0; i < vectors.size(); ++i) {
            const Vec3& orig = vectors[i].first;
            const Vec3& dir = vectors[i].second;
            const Vec3& color = colors[i];
            renderer.drawVector(orig, dir, color);
            renderer.drawPoint({orig.x + dir.x, orig.y + dir.y, 0.0f}, color);
        }

        renderer.endFrame();
    }

    return 0;
}
