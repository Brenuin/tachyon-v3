#include "collision/broadphase/grid.h"
#include <cmath>

namespace tachyon {

BroadphaseGrid::BroadphaseGrid(real cellSize)
    : cellSize(cellSize), invCellSize(1.0f / cellSize) {}

void BroadphaseGrid::clear() {
    grid.clear();
}

void BroadphaseGrid::insert(RigidBody* obj) {
    real r = obj->boundingRadius;

    GridCoord min = getCellCoord(obj->pos - Vec3(r, r, r));
    GridCoord max = getCellCoord(obj->pos + Vec3(r, r, r));

    for (int x = min.x; x <= max.x; ++x) {
        for (int y = min.y; y <= max.y; ++y) {
            for (int z = min.z; z <= max.z; ++z) {
                grid[{x, y, z}].push_back(obj);
            }
        }
    }
}

std::vector<std::pair<RigidBody*, RigidBody*>> BroadphaseGrid::computePotentialPairs() {
    std::vector<std::pair<RigidBody*, RigidBody*>> pairs;

    for (const auto& [cell, objects] : grid) {
        for (size_t i = 0; i < objects.size(); ++i) {
            for (size_t j = i + 1; j < objects.size(); ++j) {
                pairs.emplace_back(objects[i], objects[j]);
            }
        }
    }

    return pairs;
}

GridCoord BroadphaseGrid::getCellCoord(const Vec3& pos) const {
    return {
        static_cast<int>(std::floor(pos.x * invCellSize)),
        static_cast<int>(std::floor(pos.y * invCellSize)),
        static_cast<int>(std::floor(pos.z * invCellSize))
    };
}

} // namespace tachyon
