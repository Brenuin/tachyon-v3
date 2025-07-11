#pragma once

#include <unordered_map>
#include <vector>
#include <cmath>
#include <utility>
#include <functional>

#include "objects/rigid/rigid_body.h"
#include "utilities/precision.h"

namespace tachyon {

struct GridCoord {
    int x, y, z;

    bool operator==(const GridCoord& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

} 

// Hash function for GridCoord so it can be used in std::unordered_map
namespace std {
    template <>
    struct hash<tachyon::GridCoord> {
        size_t operator()(const tachyon::GridCoord& c) const {
            return ((c.x * 73856093) ^ (c.y * 19349663) ^ (c.z * 83492791));
        }
    };
}

namespace tachyon {

class BroadphaseGrid {
public:
    BroadphaseGrid(real cellSize);

    void clear();

    void insert(RigidBody* obj);

    std::vector<std::pair<RigidBody*, RigidBody*>> computePotentialPairs();

private:
    real cellSize;
    real invCellSize;

    std::unordered_map<GridCoord, std::vector<RigidBody*>> grid;

    GridCoord getCellCoord(const Vec3& pos) const;
};

} 