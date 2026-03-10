#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>

class DescEditServiceTests : public ::testing::Test
{
protected:
    DescEditService const& _service = DescEditService::get();

    bool hasAnyCrossingConnections(Desc const& description) const
    {
        auto cache = description.createCache();

        // Collect all connection line segments as pairs of positions
        struct Segment
        {
            RealVector2D start;
            RealVector2D end;
        };
        std::vector<Segment> segments;

        for (auto const& object : description._objects) {
            for (auto const& connection : object._connections) {
                auto const& connectedObject = description.getObjectRef(connection._objectId, cache);
                segments.push_back({object._pos, connectedObject._pos});
            }
        }

        // Check all pairs of segments for crossings
        for (size_t i = 0; i < segments.size(); ++i) {
            for (size_t j = i + 1; j < segments.size(); ++j) {
                if (Math::isCrossing(segments[i].start, segments[i].end, segments[j].start, segments[j].end)) {
                    return true;
                }
            }
        }
        return false;
    }
};

TEST_F(DescEditServiceTests, reconnectObjects_noCrossingConnections)
{
    // Arrange: 4 objects in a square where diagonals would cross
    Desc desc;
    desc._objects.emplace_back(ObjectDesc().pos({0.0f, 0.0f}).type(StructureDesc()));
    desc._objects.emplace_back(ObjectDesc().pos({1.0f, 0.0f}).type(StructureDesc()));
    desc._objects.emplace_back(ObjectDesc().pos({0.0f, 1.0f}).type(StructureDesc()));
    desc._objects.emplace_back(ObjectDesc().pos({1.0f, 1.0f}).type(StructureDesc()));

    // Act: reconnect with distance that includes diagonals (sqrt(2) ~ 1.414)
    _service.reconnectObjects(desc, 1.5f);

    // Assert: no crossing connections should exist
    EXPECT_FALSE(hasAnyCrossingConnections(desc));
}

TEST_F(DescEditServiceTests, reconnectObjects_noCrossingConnections_largerGrid)
{
    // Arrange: 3x3 grid of objects
    Desc desc;
    for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
            desc._objects.emplace_back(ObjectDesc().pos({toFloat(x), toFloat(y)}).type(StructureDesc()));
        }
    }

    // Act: reconnect with distance that includes diagonals
    _service.reconnectObjects(desc, 1.5f);

    // Assert: no crossing connections should exist
    EXPECT_FALSE(hasAnyCrossingConnections(desc));
}

TEST_F(DescEditServiceTests, reconnectObjects_adjacentConnectionsStillCreated)
{
    // Arrange: 4 objects in a square
    Desc desc;
    desc._objects.emplace_back(ObjectDesc().pos({0.0f, 0.0f}).type(StructureDesc()));
    desc._objects.emplace_back(ObjectDesc().pos({1.0f, 0.0f}).type(StructureDesc()));
    desc._objects.emplace_back(ObjectDesc().pos({0.0f, 1.0f}).type(StructureDesc()));
    desc._objects.emplace_back(ObjectDesc().pos({1.0f, 1.0f}).type(StructureDesc()));

    // Act: reconnect with distance that only includes adjacent (not diagonals)
    _service.reconnectObjects(desc, 1.05f);

    // Assert: each corner object should have exactly 2 connections (to its adjacent neighbors)
    for (auto const& object : desc._objects) {
        EXPECT_EQ(2, object._connections.size());
    }

    // Assert: no crossing connections
    EXPECT_FALSE(hasAnyCrossingConnections(desc));
}
