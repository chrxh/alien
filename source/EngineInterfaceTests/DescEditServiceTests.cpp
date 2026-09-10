#include <ranges>

#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/DescEditService.h>
#include <EngineInterface/Descs.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SpaceCalculator.h>

class DescEditServiceTests : public ::testing::Test
{
protected:
    DescEditService const& _service = DescEditService::get();

    std::vector<RealVector2D> getSortedObjectPositions(ContentDesc const& content) const
    {
        std::vector<RealVector2D> result;
        for (auto const& object : content._objects) {
            result.emplace_back(object._pos);
        }
        std::ranges::sort(result);
        return result;
    }

    std::vector<RealVector2D> getSortedEnergyPositions(ContentDesc const& content) const
    {
        std::vector<RealVector2D> result;
        for (auto const& energy : content._energies) {
            result.emplace_back(energy._pos);
        }
        std::ranges::sort(result);
        return result;
    }

    bool hasOnlyResolvableConnections(ContentDesc const& content) const
    {
        std::unordered_set<uint64_t> objectIds;
        for (auto const& object : content._objects) {
            objectIds.insert(object._id);
        }
        for (auto const& object : content._objects) {
            for (auto const& connection : object._connections) {
                if (!objectIds.contains(connection._objectId)) {
                    return false;
                }
            }
        }
        return true;
    }

    bool hasSymmetricConnections(ContentDesc const& content) const
    {
        auto cache = content.createCache();
        for (auto const& object : content._objects) {
            for (auto const& connection : object._connections) {
                if (!content.getObjectRef(connection._objectId, cache).isConnectedTo(object._id)) {
                    return false;
                }
            }
        }
        return true;
    }

    bool haveConnectionAnglesFullTurn(ContentDesc const& content) const
    {
        for (auto const& object : content._objects) {
            if (object._connections.empty()) {
                continue;
            }
            auto sumAngles = 0.0f;
            for (auto const& connection : object._connections) {
                sumAngles += connection._angleFromPrevious;
            }
            if (std::abs(sumAngles - 360.0f) > 0.001f) {
                return false;
            }
        }
        return true;
    }

    bool haveConnectionDistancesTorusLength(ContentDesc const& content, IntVector2D const& worldSize) const
    {
        SpaceCalculator space(worldSize);
        auto cache = content.createCache();
        for (auto const& object : content._objects) {
            for (auto const& connection : object._connections) {
                auto const& connectedObject = content.getObjectRef(connection._objectId, cache);
                if (std::abs(space.distance(object._pos, connectedObject._pos) - connection._distance) > 0.001f) {
                    return false;
                }
            }
        }
        return true;
    }

    bool hasAnyCrossingConnections(ContentDesc const& description) const
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
    auto desc = ContentDesc().objects({
        ObjectDesc().pos({0.0f, 0.0f}).type(SolidDesc()),
        ObjectDesc().pos({1.0f, 0.0f}).type(SolidDesc()),
        ObjectDesc().pos({0.0f, 1.0f}).type(SolidDesc()),
        ObjectDesc().pos({1.0f, 1.0f}).type(SolidDesc()),
    });

    // Act: reconnect with distance that includes diagonals (sqrt(2) ~ 1.414)
    _service.reconnectObjects(desc, 1.5f);

    // Assert: no crossing connections should exist
    EXPECT_FALSE(hasAnyCrossingConnections(desc));
}

TEST_F(DescEditServiceTests, reconnectObjects_noCrossingConnections_largerGrid)
{
    // Arrange: 3x3 grid of objects
    auto desc = ContentDesc().objects({
        ObjectDesc().pos({0.0f, 0.0f}).type(SolidDesc()),
        ObjectDesc().pos({1.0f, 0.0f}).type(SolidDesc()),
        ObjectDesc().pos({2.0f, 0.0f}).type(SolidDesc()),
        ObjectDesc().pos({0.0f, 1.0f}).type(SolidDesc()),
        ObjectDesc().pos({1.0f, 1.0f}).type(SolidDesc()),
        ObjectDesc().pos({2.0f, 1.0f}).type(SolidDesc()),
        ObjectDesc().pos({0.0f, 2.0f}).type(SolidDesc()),
        ObjectDesc().pos({1.0f, 2.0f}).type(SolidDesc()),
        ObjectDesc().pos({2.0f, 2.0f}).type(SolidDesc()),
    });

    // Act: reconnect with distance that includes diagonals
    _service.reconnectObjects(desc, 1.5f);

    // Assert: no crossing connections should exist
    EXPECT_FALSE(hasAnyCrossingConnections(desc));
}

TEST_F(DescEditServiceTests, reconnectObjects_adjacentConnectionsStillCreated)
{
    // Arrange: 4 objects in a square
    auto desc = ContentDesc().objects({
        ObjectDesc().pos({0.0f, 0.0f}).type(SolidDesc()),
        ObjectDesc().pos({1.0f, 0.0f}).type(SolidDesc()),
        ObjectDesc().pos({0.0f, 1.0f}).type(SolidDesc()),
        ObjectDesc().pos({1.0f, 1.0f}).type(SolidDesc()),
    });

    // Act: reconnect with distance that only includes adjacent (not diagonals)
    _service.reconnectObjects(desc, 1.05f);

    // Assert: each corner object should have exactly 2 connections (to its adjacent neighbors)
    for (auto const& object : desc._objects) {
        EXPECT_EQ(2, object._connections.size());
    }

    // Assert: no crossing connections
    EXPECT_FALSE(hasAnyCrossingConnections(desc));
}

TEST_F(DescEditServiceTests, scaleContent_unchangedSize_contentUntouched)
{
    auto desc = ContentDesc().objects({ObjectDesc().id(1).pos({10.0f, 20.0f}).type(SolidDesc())}).energies({EnergyDesc().id(2).pos({30.0f, 40.0f})});
    auto origDesc = desc;

    _service.scaleContent(desc, {100, 100}, {100, 100});

    EXPECT_TRUE(origDesc == desc);
}

TEST_F(DescEditServiceTests, scaleContent_doubledWidth_contentDuplicatedHorizontally)
{
    auto desc = ContentDesc().objects({ObjectDesc().id(1).pos({10.0f, 20.0f}).type(SolidDesc())});

    _service.scaleContent(desc, {100, 100}, {200, 100});

    EXPECT_EQ(std::vector<RealVector2D>({{10.0f, 20.0f}, {110.0f, 20.0f}}), getSortedObjectPositions(desc));
}

TEST_F(DescEditServiceTests, scaleContent_doubledWidthAndHeight_contentDuplicatedFourTimes)
{
    auto desc = ContentDesc().objects({ObjectDesc().id(1).pos({10.0f, 20.0f}).type(SolidDesc())});

    _service.scaleContent(desc, {100, 100}, {200, 200});

    EXPECT_EQ(std::vector<RealVector2D>({{10.0f, 20.0f}, {10.0f, 120.0f}, {110.0f, 20.0f}, {110.0f, 120.0f}}), getSortedObjectPositions(desc));
}

TEST_F(DescEditServiceTests, scaleContent_reducedSize_protrudingObjectsRemoved)
{
    auto desc = ContentDesc().objects({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(SolidDesc()),
        ObjectDesc().id(2).pos({80.0f, 10.0f}).type(SolidDesc()),
        ObjectDesc().id(3).pos({10.0f, 80.0f}).type(SolidDesc()),
        ObjectDesc().id(4).pos({80.0f, 80.0f}).type(SolidDesc()),
    });

    _service.scaleContent(desc, {100, 100}, {50, 50});

    EXPECT_EQ(std::vector<RealVector2D>({{10.0f, 10.0f}}), getSortedObjectPositions(desc));
}

TEST_F(DescEditServiceTests, scaleContent_widerAndLowerSize_widthDuplicatedAndHeightClipped)
{
    auto desc = ContentDesc().objects({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(SolidDesc()),
        ObjectDesc().id(2).pos({10.0f, 80.0f}).type(SolidDesc()),
    });

    _service.scaleContent(desc, {100, 100}, {200, 50});

    EXPECT_EQ(std::vector<RealVector2D>({{10.0f, 10.0f}, {110.0f, 10.0f}}), getSortedObjectPositions(desc));
}

TEST_F(DescEditServiceTests, scaleContent_partiallyCoveredTile_objectsBeyondNewBorderRemoved)
{
    auto desc = ContentDesc().objects({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(SolidDesc()),
        ObjectDesc().id(2).pos({60.0f, 10.0f}).type(SolidDesc()),
    });

    _service.scaleContent(desc, {100, 100}, {150, 100});

    EXPECT_EQ(std::vector<RealVector2D>({{10.0f, 10.0f}, {60.0f, 10.0f}, {110.0f, 10.0f}}), getSortedObjectPositions(desc));
}

TEST_F(DescEditServiceTests, scaleContent_doubledWidth_energiesDuplicated)
{
    auto desc = ContentDesc().energies({EnergyDesc().id(1).pos({10.0f, 20.0f}).energy(50.0f)});

    _service.scaleContent(desc, {100, 100}, {200, 100});

    EXPECT_EQ(std::vector<RealVector2D>({{10.0f, 20.0f}, {110.0f, 20.0f}}), getSortedEnergyPositions(desc));
    for (auto const& energy : desc._energies) {
        EXPECT_EQ(50.0f, energy._energy);
    }
}

TEST_F(DescEditServiceTests, scaleContent_reducedSize_protrudingEnergiesRemoved)
{
    auto desc = ContentDesc().energies({
        EnergyDesc().id(1).pos({10.0f, 10.0f}),
        EnergyDesc().id(2).pos({80.0f, 10.0f}),
    });

    _service.scaleContent(desc, {100, 100}, {50, 100});

    EXPECT_EQ(std::vector<RealVector2D>({{10.0f, 10.0f}}), getSortedEnergyPositions(desc));
}

TEST_F(DescEditServiceTests, scaleContent_duplicatedCreature_entitiesGetUniqueIds)
{
    auto desc = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}),
        ObjectDesc().id(2).pos({11.0f, 10.0f}),
    });
    desc.addConnection(1, 2);

    _service.scaleContent(desc, {100, 100}, {200, 200});

    EXPECT_EQ(8, desc._objects.size());
    EXPECT_EQ(4, desc._creatures.size());
    EXPECT_EQ(4, desc._genomes.size());
    EXPECT_TRUE(desc.hasUniqueIds());
    EXPECT_TRUE(hasOnlyResolvableConnections(desc));
    EXPECT_TRUE(hasSymmetricConnections(desc));
}

TEST_F(DescEditServiceTests, scaleContent_duplicatedCreature_lineageIdsPreserved)
{
    auto desc = ContentDesc().addCreature({ObjectDesc().id(1).pos({10.0f, 10.0f})}, CreatureDesc().lineageId(7).generation(3));

    _service.scaleContent(desc, {100, 100}, {200, 100});

    ASSERT_EQ(2, desc._creatures.size());
    for (auto const& creature : desc._creatures) {
        EXPECT_EQ(7, creature._lineageId);
        EXPECT_EQ(3, creature._generation);
    }
}

TEST_F(DescEditServiceTests, scaleContent_removedObject_noLongerReferenced)
{
    auto desc = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}),
        ObjectDesc().id(2).pos({60.0f, 10.0f}),
    });
    desc.addConnection(1, 2);

    _service.scaleContent(desc, {100, 100}, {50, 100});

    ASSERT_EQ(1, desc._objects.size());
    EXPECT_TRUE(desc._objects.front()._connections.empty());
    EXPECT_TRUE(hasOnlyResolvableConnections(desc));
}

TEST_F(DescEditServiceTests, scaleContent_removedObject_remainingConnectionAnglesCompleteFullTurn)
{
    auto desc = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({48.0f, 50.0f}),
        ObjectDesc().id(2).pos({47.0f, 50.0f}),
        ObjectDesc().id(3).pos({49.0f, 50.0f}),
        ObjectDesc().id(4).pos({48.0f, 49.0f}),
        ObjectDesc().id(5).pos({48.0f, 51.0f}),
    });
    desc.addConnection(1, 2);
    desc.addConnection(1, 3);
    desc.addConnection(1, 4);
    desc.addConnection(1, 5);

    _service.scaleContent(desc, {100, 100}, {49, 100});

    ASSERT_EQ(4, desc._objects.size());
    EXPECT_EQ(3, desc.getObjectRef(1)._connections.size());
    EXPECT_TRUE(haveConnectionAnglesFullTurn(desc));
    EXPECT_TRUE(hasOnlyResolvableConnections(desc));
    EXPECT_TRUE(hasSymmetricConnections(desc));
}

TEST_F(DescEditServiceTests, scaleContent_creatureLosingAllCells_creatureAndGenomeRemoved)
{
    auto desc = ContentDesc().addCreature({ObjectDesc().id(1).pos({10.0f, 10.0f})}).addCreature({ObjectDesc().id(2).pos({80.0f, 10.0f})});

    _service.scaleContent(desc, {100, 100}, {50, 100});

    ASSERT_EQ(1, desc._objects.size());
    EXPECT_EQ(1, desc._creatures.size());
    EXPECT_EQ(1, desc._genomes.size());
    EXPECT_EQ(desc._creatures.front()._genomeId, desc._genomes.front()._id);
    EXPECT_EQ(1, desc._creatures.front()._numCells);
}

TEST_F(DescEditServiceTests, scaleContent_creatureLosingSomeCells_numCellsUpdated)
{
    auto desc = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}),
        ObjectDesc().id(2).pos({11.0f, 10.0f}),
        ObjectDesc().id(3).pos({80.0f, 10.0f}),
    });
    desc.addConnection(1, 2);

    _service.scaleContent(desc, {100, 100}, {50, 100});

    ASSERT_EQ(1, desc._creatures.size());
    EXPECT_EQ(2, desc._creatures.front()._numCells);
}

TEST_F(DescEditServiceTests, scaleContent_removedObject_lastConstructedCellIdReset)
{
    auto desc = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().constructor(ConstructorDesc().lastConstructedCellId(uint64_t(2)))),
        ObjectDesc().id(2).pos({80.0f, 10.0f}),
    });

    _service.scaleContent(desc, {100, 100}, {50, 100});

    ASSERT_EQ(1, desc._objects.size());
    EXPECT_FALSE(desc._objects.front().getCellRef()._constructor->_lastConstructedCellId.has_value());
}

TEST_F(DescEditServiceTests, scaleContent_creatureAcrossTorusBorder_duplicatesStayConnected)
{
    auto desc = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({99.0f, 50.0f}),
        ObjectDesc().id(2).pos({0.0f, 50.0f}),
    });
    desc.addConnection(1, 2, RealVector2D{100.0f, 50.0f});

    _service.scaleContent(desc, {100, 100}, {200, 100});

    EXPECT_EQ(std::vector<RealVector2D>({{0.0f, 50.0f}, {99.0f, 50.0f}, {100.0f, 50.0f}, {199.0f, 50.0f}}), getSortedObjectPositions(desc));
    for (auto const& object : desc._objects) {
        EXPECT_EQ(1, object._connections.size());
    }
    EXPECT_TRUE(hasSymmetricConnections(desc));
    EXPECT_TRUE(haveConnectionDistancesTorusLength(desc, {200, 100}));
}

TEST_F(DescEditServiceTests, scaleContent_creatureAcrossTorusBorder_reducedSizeKeepsRemainderConnected)
{
    auto desc = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({98.0f, 20.0f}),
        ObjectDesc().id(2).pos({99.0f, 20.0f}),
        ObjectDesc().id(3).pos({0.0f, 20.0f}),
        ObjectDesc().id(4).pos({0.0f, 80.0f}),
    });
    desc.addConnection(1, 2);
    desc.addConnection(2, 3, RealVector2D{100.0f, 20.0f});
    desc.addConnection(3, 4);

    _service.scaleContent(desc, {100, 100}, {100, 50});

    ASSERT_EQ(3, desc._objects.size());
    EXPECT_EQ(1, desc.getObjectRef(3)._connections.size());
    EXPECT_TRUE(haveConnectionAnglesFullTurn(desc));
    EXPECT_TRUE(hasOnlyResolvableConnections(desc));
    EXPECT_TRUE(hasSymmetricConnections(desc));
    EXPECT_TRUE(haveConnectionDistancesTorusLength(desc, {100, 50}));
}

TEST_F(DescEditServiceTests, scaleContent_nonCreatureObjectsAcrossTorusBorder_stayConnected)
{
    auto desc = ContentDesc().objects({
        ObjectDesc().id(1).pos({99.0f, 50.0f}).type(SolidDesc()),
        ObjectDesc().id(2).pos({0.0f, 50.0f}).type(SolidDesc()),
    });
    desc.addConnection(1, 2, RealVector2D{100.0f, 50.0f});

    _service.scaleContent(desc, {100, 100}, {200, 200});

    EXPECT_EQ(8, desc._objects.size());
    EXPECT_TRUE(hasSymmetricConnections(desc));
    EXPECT_TRUE(haveConnectionDistancesTorusLength(desc, {200, 200}));
}

TEST_F(DescEditServiceTests, scaleContent_allObjectsInsideNewWorld)
{
    auto desc = ContentDesc().objects({
        ObjectDesc().id(1).pos({99.0f, 50.0f}).type(SolidDesc()),
        ObjectDesc().id(2).pos({0.0f, 50.0f}).type(SolidDesc()),
    });
    desc.addConnection(1, 2, RealVector2D{100.0f, 50.0f});

    _service.scaleContent(desc, {100, 100}, {250, 130});

    EXPECT_EQ(5, desc._objects.size());
    for (auto const& object : desc._objects) {
        EXPECT_GE(object._pos.x, 0.0f);
        EXPECT_GE(object._pos.y, 0.0f);
        EXPECT_LT(object._pos.x, 250.0f);
        EXPECT_LT(object._pos.y, 130.0f);
    }
}

TEST_F(DescEditServiceTests, scaleContent_idGeneratorBehindExistingIds_duplicatesGetUniqueIds)
{
    auto desc = ContentDesc()
                    .objects({
                        ObjectDesc().id(0).pos({10.0f, 10.0f}).type(SolidDesc()),
                        ObjectDesc().id(1).pos({20.0f, 10.0f}).type(SolidDesc()),
                    })
                    .energies({EnergyDesc().id(2).pos({30.0f, 10.0f})});
    NumberGenerator::get().setIds(Ids());

    _service.scaleContent(desc, {100, 100}, {200, 100});

    EXPECT_EQ(4, desc._objects.size());
    EXPECT_EQ(2, desc._energies.size());
    EXPECT_TRUE(desc.hasUniqueIds());
}

TEST_F(DescEditServiceTests, scaleContent_repeatedWidening_duplicatesGetUniqueIds)
{
    auto desc = ContentDesc().objects({
        ObjectDesc().id(5).pos({0.5f, 10.0f}).type(SolidDesc()),
        ObjectDesc().id(6).pos({50.0f, 10.0f}).type(SolidDesc()),
    });

    NumberGenerator::get().setIds(Ids());
    _service.scaleContent(desc, {100, 100}, {101, 100});
    ASSERT_TRUE(desc.hasUniqueIds());

    NumberGenerator::get().setIds(Ids());
    _service.scaleContent(desc, {101, 100}, {200, 100});

    EXPECT_TRUE(desc.hasUniqueIds());
}
