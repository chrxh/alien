#include <algorithm>
#include <map>
#include <random>
#include <set>
#include <numeric>

#include <gtest/gtest.h>

#include <Base/Math.h>
#include <Data/Descs.h>
#include <PersisterInterface/SerializerService.h>

namespace
{
    auto constexpr WORLD_SIZE = 2000;
    auto constexpr GRID_SPACING = 200;
    auto constexpr GRID_OFFSET = GRID_SPACING / 2;
    auto constexpr NUM_SQUARES_PER_AXIS = WORLD_SIZE / GRID_SPACING;
    auto constexpr OPENING_HALF_LENGTH = 5;
    auto constexpr RANDOM_SEED = 42;
}

TEST(GridSimulationGenerator, DISABLED_createSolidGrid)
{
    auto isGridLine = [](int coord) { return coord % GRID_SPACING == GRID_OFFSET; };
    auto wrap = [](int coord) { return (coord + WORLD_SIZE) % WORLD_SIZE; };
    auto rodLine = [](int index) { return GRID_OFFSET + index * GRID_SPACING; };
    auto squareCenter = [&](int index) { return wrap(rodLine(index) + GRID_SPACING / 2); };

    struct Wall
    {
        int square1;
        int square2;
        std::pair<int, int> center;
    };
    std::vector<Wall> walls;
    for (auto row = 0; row < NUM_SQUARES_PER_AXIS; ++row) {
        for (auto column = 0; column < NUM_SQUARES_PER_AXIS; ++column) {
            auto nextColumn = (column + 1) % NUM_SQUARES_PER_AXIS;
            auto nextRow = (row + 1) % NUM_SQUARES_PER_AXIS;
            auto square = column + row * NUM_SQUARES_PER_AXIS;
            walls.emplace_back(square, nextColumn + row * NUM_SQUARES_PER_AXIS, std::pair{rodLine(nextColumn), squareCenter(row)});
            walls.emplace_back(square, column + nextRow * NUM_SQUARES_PER_AXIS, std::pair{squareCenter(column), rodLine(nextRow)});
        }
    }
    std::mt19937 randomEngine(RANDOM_SEED);
    std::ranges::shuffle(walls, randomEngine);

    // Random spanning tree over the squares (Kruskal)
    std::vector<int> parent(NUM_SQUARES_PER_AXIS * NUM_SQUARES_PER_AXIS);
    std::iota(parent.begin(), parent.end(), 0);
    auto findRoot = [&](int square) {
        while (parent.at(square) != square) {
            parent.at(square) = parent.at(parent.at(square));
            square = parent.at(square);
        }
        return square;
    };
    std::vector<std::pair<int, int>> openingCenters;
    for (auto const& wall : walls) {
        auto root1 = findRoot(wall.square1);
        auto root2 = findRoot(wall.square2);
        if (root1 != root2) {
            parent.at(root1) = root2;
            openingCenters.emplace_back(wall.center);
        }
    }

    std::set<std::pair<int, int>> removedPositions;
    std::set<std::pair<int, int>> openingBorderPositions;
    for (auto const& [centerX, centerY] : openingCenters) {
        auto alongX = isGridLine(centerY);
        for (auto offset = -OPENING_HALF_LENGTH; offset <= OPENING_HALF_LENGTH; ++offset) {
            auto position = alongX ? std::pair{wrap(centerX + offset), centerY} : std::pair{centerX, wrap(centerY + offset)};
            if (std::abs(offset) == OPENING_HALF_LENGTH) {
                openingBorderPositions.insert(position);
            } else {
                removedPositions.insert(position);
            }
        }
    }

    std::map<std::pair<int, int>, ObjectDesc> objectByPos;
    for (auto line = GRID_OFFSET; line < WORLD_SIZE; line += GRID_SPACING) {
        for (auto pos = 0; pos < WORLD_SIZE; ++pos) {
            for (auto const& position : {std::pair{pos, line}, std::pair{line, pos}}) {
                if (!objectByPos.contains(position) && !removedPositions.contains(position)) {
                    auto const& [x, y] = position;
                    auto isStatic = (isGridLine(x) && isGridLine(y)) || openingBorderPositions.contains(position);
                    objectByPos.emplace(position, ObjectDesc().pos({toFloat(x), toFloat(y)}).type(SolidDesc()).isStatic(isStatic));
                }
            }
        }
    }

    ContentDesc content;
    for (auto& [pos, object] : objectByPos) {
        auto const& [x, y] = pos;
        std::vector<std::pair<int, int>> directions;
        if (isGridLine(y)) {
            directions.emplace_back(1, 0);
            directions.emplace_back(-1, 0);
        }
        if (isGridLine(x)) {
            directions.emplace_back(0, 1);
            directions.emplace_back(0, -1);
        }
        std::erase_if(directions, [&](auto const& dir) { return !objectByPos.contains({wrap(x + dir.first), wrap(y + dir.second)}); });
        std::ranges::sort(directions, {}, [](auto const& dir) { return Math::angleOfVector(RealVector2D{toFloat(dir.first), toFloat(dir.second)}); });

        auto angleSpan = 360.0f / toFloat(directions.size());
        for (auto const& [dirX, dirY] : directions) {
            auto const& neighbor = objectByPos.at({wrap(x + dirX), wrap(y + dirY)});
            object._connections.emplace_back(ConnectionDesc().objectId(neighbor._id).distance(1.0f).angleFromPrevious(angleSpan));
        }
        content._objects.emplace_back(object);
    }

    auto numRodObjects = 2 * NUM_SQUARES_PER_AXIS * WORLD_SIZE - NUM_SQUARES_PER_AXIS * NUM_SQUARES_PER_AXIS;
    auto numOpenings = NUM_SQUARES_PER_AXIS * NUM_SQUARES_PER_AXIS - 1;
    EXPECT_EQ(numOpenings, openingCenters.size());
    EXPECT_EQ(numRodObjects - numOpenings * (2 * OPENING_HALF_LENGTH - 1), content._objects.size());
    EXPECT_EQ(NUM_SQUARES_PER_AXIS * NUM_SQUARES_PER_AXIS + 2 * numOpenings, std::ranges::count_if(content._objects, [](auto const& object) {
                  return object._isStatic;
              }));

    auto simulation =
        SimulationDesc().worldSize({WORLD_SIZE, WORLD_SIZE}).center({toFloat(WORLD_SIZE) / 2, toFloat(WORLD_SIZE) / 2}).zoom(1.0f).mainData(content);
    EXPECT_TRUE(SerializerService::get().serializeSimulationToFiles("Grid.sim", simulation));
}
