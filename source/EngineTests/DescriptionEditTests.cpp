#include <algorithm>
#include <chrono>
#include <ranges>

#include <gtest/gtest.h>

#include <Base/Definitions.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class DescriptionEditTests : public IntegrationTestFramework
{
public:
    DescriptionEditTests()
        : IntegrationTestFramework({100, 100})
    {}
    virtual ~DescriptionEditTests() = default;

protected:
    bool areAngelsCorrect(Description const& data) const
    {
        for (auto const& object : data._objects) {
            if (!object._connections.empty()) {
                float sumAngles = 0;
                for (auto const& connection : object._connections) {
                    sumAngles += connection._angleFromPrevious;
                }
                if (std::abs(sumAngles - 360.0f) > NEAR_ZERO) {
                    return false;
                }
            }
        }
        return true;
    }
};


TEST_F(DescriptionEditTests, correctConnections)
{
    auto origData = DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().width(10).height(10).center({50.0f, 99.0f}));
    _simulationFacade->setSimulationData(origData);

    auto data = _simulationFacade->getSimulationData();

    DescriptionEditService::get().duplicate(data, {100, 100}, {100, 100});

    EXPECT_TRUE(areAngelsCorrect(data));
}


TEST_F(DescriptionEditTests, addThirdConnection1)
{
    auto data = Description().objects({
        ObjectDescription().id(1).pos({0, 0}),
        ObjectDescription().id(2).pos({1, 0}),
        ObjectDescription().id(3).pos({0, 1}),
        ObjectDescription().id(4).pos({0, -1}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(1, 4);

    auto object = data.getObjectRef(1);

    EXPECT_EQ(3, object._connections.size());

    auto connection1 = object._connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection1._angleFromPrevious));

    auto connection2 = object._connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2._angleFromPrevious));

    auto connection3 = object._connections.at(2);
    EXPECT_TRUE(approxCompare(1.0f, connection3._distance));
    EXPECT_TRUE(approxCompare(180.0f, connection3._angleFromPrevious));
}

TEST_F(DescriptionEditTests, addThirdConnection2)
{
    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({0, 0}),
        ObjectDescription().id(2).pos({1, 0}),
        ObjectDescription().id(3).pos({-1, 0}),
        ObjectDescription().id(4).pos({0, 1}),
    };
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(1, 4);

    auto object = data.getObjectRef(1);

    EXPECT_EQ(3, object._connections.size());

    auto connection1 = object._connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1._distance));
    EXPECT_TRUE(approxCompare(180.0f, connection1._angleFromPrevious));

    auto connection2 = object._connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2._angleFromPrevious));

    auto connection3 = object._connections.at(2);
    EXPECT_TRUE(approxCompare(1.0f, connection3._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection3._angleFromPrevious));
}


enum class CellsOnCreature
{
    No,
    Yes
};
class DescriptionEditTests_CellIdGeneration
    : public DescriptionEditTests
    , public testing::WithParamInterface<CellsOnCreature>
{};

INSTANTIATE_TEST_SUITE_P(
    DescriptionEditTests_CellIdGeneration,
    DescriptionEditTests_CellIdGeneration,
    ::testing::Values(CellsOnCreature::No, CellsOnCreature::Yes));


TEST_P(DescriptionEditTests_CellIdGeneration, assignNewIds_differentCellIds)
{
    // Create test data
    Description data;
    if (GetParam() == CellsOnCreature::No) {
        data._objects = {ObjectDescription().id(0), ObjectDescription().id(1)};
    } else {
        data.addCreature({ObjectDescription().id(0), ObjectDescription().id(1)}, CreatureDescription());
    }

    // Perform action
    data.assignNewIds();

    // Check result
    std::unordered_set<uint64_t> ids;
    for (auto const& object : data._objects) { ids.insert(object._id); }

    EXPECT_EQ(2, ids.size());
}

TEST_P(DescriptionEditTests_CellIdGeneration, assignNewIds_sameCellIds)
{
    // Create test data
    auto createCollection = [] {
        if (GetParam() == CellsOnCreature::No) {
            return Description().objects({ObjectDescription().id(0), ObjectDescription().id(0)});
        } else {
            return Description().addCreature({ObjectDescription().id(0), ObjectDescription().id(0)}, CreatureDescription());
        }
    };
    auto data = createCollection();

    // Perform action
    data.assignNewIds();

    // Check result
    std::unordered_set<uint64_t> ids;
    for (auto const& object : data._objects) { ids.insert(object._id); }
    EXPECT_EQ(2, ids.size());
}

TEST_P(DescriptionEditTests_CellIdGeneration, assignNewIds_preserveOrder)
{
    // Create test data
    Description data;
    if (GetParam() == CellsOnCreature::No) {
        for (int i = 0; i < 10; ++i) {
            data._objects.emplace_back(ObjectDescription().id(i).type(CellDescription().age(i)));
        }
        std::sort(data._objects.begin(), data._objects.end(), [](auto const& lhs, auto const& rhs) { return lhs._id > rhs._id; });
    } else {
        std::vector<ObjectDescription> cells;
        for (int i = 0; i < 10; ++i) {
            cells.emplace_back(ObjectDescription().id(i).type(CellDescription().age(i)));
        }
        std::sort(cells.begin(), cells.end(), [](auto const& lhs, auto const& rhs) { return lhs._id > rhs._id; });
        data.addCreature(cells, CreatureDescription());
    }

    // Perform action
    data.assignNewIds();

    // Check result
    auto cells = GetParam() == CellsOnCreature::No ? data._objects : data.getObjectsForCreature(data._creatures.front()._id);
    std::map<int, ObjectDescription> ageToCell;
    for (auto const& object : cells) {
        ageToCell.insert_or_assign(object.getCellRef()._age, object);
    }
    EXPECT_EQ(10, ageToCell.size());

    std::optional<uint64_t> lastObjectId;
    for (auto const& object : ageToCell | std::views::values) {
        if (!lastObjectId.has_value()) {
            lastObjectId = object._id;
        } else {
            EXPECT_TRUE(lastObjectId.value() < object._id);
        }
    }
}

TEST_F(DescriptionEditTests, assignNewIds_sameConnectionOnDifferentCreatures)
{
    // Create test data
    auto data = Description()
                    .objects({
                        ObjectDescription().id(0).connections({ConnectionDescription().objectId(1)}),
                        ObjectDescription().id(1).connections({ConnectionDescription().objectId(0)}),
                    })
                    .addCreature({
                        ObjectDescription().id(0).connections({ConnectionDescription().objectId(1)}),
                        ObjectDescription().id(1).connections({ConnectionDescription().objectId(0)}),
                    }, CreatureDescription());

    // Perform action
    data.assignNewIds();

    // Check result
    std::unordered_set<uint64_t> ids;
    for (auto const& object : data._objects) { ids.insert(object._id); }
    ASSERT_EQ(4, ids.size());

    ASSERT_EQ(2, data.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, data._creatures.size());

    for (auto const& creature : data._creatures) {
        auto creatureCells = data.getObjectsForCreature(creature._id);
        ASSERT_EQ(2, creatureCells.size());
        auto const& object1 = creatureCells.front();
        auto const& object2 = creatureCells.back();

        ASSERT_EQ(1, object1._connections.size());
        EXPECT_EQ(object2._id, object1._connections.front()._objectId);

        ASSERT_EQ(1, object2._connections.size());
        EXPECT_EQ(object1._id, object2._connections.front()._objectId);
    }
}

TEST_F(DescriptionEditTests, assignNewIds_connectionBetweenCreature)
{
    // Create test data
    auto data = Description()
                    .addCreature({
                        ObjectDescription().id(0).connections({ConnectionDescription().objectId(2)}),
                        ObjectDescription().id(1),
                    }, CreatureDescription())
                    .addCreature({
                        ObjectDescription().id(2).connections({ConnectionDescription().objectId(0)}),
                    }, CreatureDescription());

    // Perform action
    data.assignNewIds();

    // Check result
    std::unordered_set<uint64_t> ids;
    for (auto const& object : data._objects) { ids.insert(object._id); }
    ASSERT_EQ(3, ids.size());

    ASSERT_EQ(0, data.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, data._creatures.size());

    std::optional<CreatureDescription> smallCreature, largeCreature;
    for (auto const& creature : data._creatures) {
        if (data.getObjectsForCreature(creature._id).size() == 1) {
            smallCreature = creature;
        }
        if (data.getObjectsForCreature(creature._id).size() == 2) {
            largeCreature = creature;
        }
    }
    ASSERT_TRUE(smallCreature.has_value());
    ASSERT_TRUE(largeCreature.has_value());

    auto smallCreatureCells = data.getObjectsForCreature(smallCreature->_id);
    EXPECT_EQ(1, smallCreatureCells.front()._connections.size());
    auto connectedObjectId = smallCreatureCells.front()._connections.front()._objectId;

    auto largeCreatureCells = data.getObjectsForCreature(largeCreature->_id);
    std::optional<ObjectDescription> cellWithoutConnection, cellWithConnection;
    for (auto const& object : largeCreatureCells) {
        if (object._connections.empty()) {
            cellWithoutConnection = object;
        } else {
            cellWithConnection = object;
        }
    }
    ASSERT_TRUE(cellWithoutConnection.has_value());
    ASSERT_TRUE(cellWithConnection.has_value());

    ASSERT_EQ(1, cellWithConnection->_connections.size());
    EXPECT_EQ(connectedObjectId, cellWithConnection->_id);
    EXPECT_EQ(smallCreatureCells.front()._id, cellWithConnection->_connections.front()._objectId);
}

TEST_F(DescriptionEditTests, assignNewIds_connectionNotContained)
{
    // Create test data
    auto data = Description()
                    .objects({
                        ObjectDescription().id(0).connections({ConnectionDescription().objectId(3)}),
                        ObjectDescription().id(1),
                    })
                    .addCreature({
                        ObjectDescription().id(2).connections({ConnectionDescription().objectId(4)}),
                    }, CreatureDescription());
    // Perform action
    data.assignNewIds();

    // Check result
    std::unordered_set<uint64_t> ids;
    for (auto const& object : data._objects) { ids.insert(object._id); }
    ASSERT_EQ(3, ids.size());

    ASSERT_EQ(2, data.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, data._creatures.size());

    // Only look at cells without creature
    std::optional<ObjectDescription> cellWithoutConnection, cellWithConnection;
    for (auto const& object : data._objects) {
        if (!object.getCellRef()._creatureId.has_value()) {
            if (object._connections.empty()) {
                cellWithoutConnection = object;
            } else {
                cellWithConnection = object;
            }
        }
    }
    ASSERT_TRUE(cellWithoutConnection.has_value());
    ASSERT_TRUE(cellWithConnection.has_value());

    ASSERT_EQ(1, cellWithConnection->_connections.size());
    EXPECT_EQ(3, cellWithConnection->_connections.front()._objectId);

    auto const& creature = data._creatures.front();
    auto creatureCells = data.getObjectsForCreature(creature._id);
    ASSERT_EQ(1, creatureCells.size());

    ASSERT_EQ(1, creatureCells.front()._connections.size());
    ASSERT_EQ(4, creatureCells.front()._connections.front()._objectId);
}

TEST_F(DescriptionEditTests, assignNewIds_cellWithLastConstructedCellId_contained)
{
    // Create test data
    auto data = Description().objects({
        ObjectDescription().id(0).type(CellDescription().cellType(ConstructorDescription().lastConstructedCellId(1))),
        ObjectDescription().id(1),
    });

    // Perform action
    data.assignNewIds();

    // Check result
    std::unordered_set<uint64_t> ids;
    for (auto const& object : data._objects) { ids.insert(object._id); }
    ASSERT_EQ(2, ids.size());

    ASSERT_EQ(2, data._objects.size());
    ASSERT_EQ(0, data._creatures.size());

    std::optional<ObjectDescription> constructor, base;
    for (auto const& object : data._objects) {
        if (object.getCellRef().getCellType() == CellType_Constructor) {
            constructor = object;
        }
        if (object.getCellRef().getCellType() == CellType_Base) {
            base = object;
        }
    }
    ASSERT_TRUE(constructor.has_value());
    ASSERT_TRUE(base.has_value());

    EXPECT_EQ(base->_id, std::get<ConstructorDescription>(constructor->getCellRef()._cellType)._lastConstructedCellId);
}

TEST_F(DescriptionEditTests, assignNewIds_cellWithLastConstructedCellId_notContained)
{
    // Create test data
    auto data = Description().objects({
        ObjectDescription().id(0).type(CellDescription().cellType(ConstructorDescription().lastConstructedCellId(2))),
        ObjectDescription().id(1),
    });

    // Perform action
    data.assignNewIds();

    // Check result
    std::unordered_set<uint64_t> ids;
    for (auto const& object : data._objects) { ids.insert(object._id); }
    ASSERT_EQ(2, ids.size());

    ASSERT_EQ(2, data._objects.size());
    ASSERT_EQ(0, data._creatures.size());

    std::optional<ObjectDescription> constructor, base;
    for (auto const& object : data._objects) {
        if (object.getCellRef().getCellType() == CellType_Constructor) {
            constructor = object;
        }
        if (object.getCellRef().getCellType() == CellType_Base) {
            base = object;
        }
    }
    ASSERT_TRUE(constructor.has_value());
    ASSERT_TRUE(base.has_value());

    EXPECT_EQ(2, std::get<ConstructorDescription>(constructor->getCellRef()._cellType)._lastConstructedCellId);
}

TEST_F(DescriptionEditTests, assignNewIds_differentParticleIds)
{
    // Create test data
    auto data = Description().energies({EnergyDescription().id(0), EnergyDescription().id(1)});

    // Perform action
    data.assignNewIds();

    // Check result
    std::unordered_set<uint64_t> ids;
    for (auto const& object : data._energies) {
        ids.insert(object._id);
    }
    EXPECT_EQ(2, ids.size());
}

TEST_F(DescriptionEditTests, assignNewIds_sameParticleIds)
{
    // Create test data
    auto data = Description().energies({EnergyDescription().id(0), EnergyDescription().id(0)});

    // Perform action
    data.assignNewIds();

    // Check result
    std::unordered_set<uint64_t> ids;
    for (auto const& object : data._energies) {
        ids.insert(object._id);
    }
    EXPECT_EQ(2, ids.size());
}

TEST_F(DescriptionEditTests, assignNewIds_differentCreatureIds)
{
    // Create test data
    auto data = Description()
                    .addCreature({ObjectDescription().id(0), ObjectDescription().id(0)}, CreatureDescription().id(0))
                    .addCreature({ObjectDescription().id(0), ObjectDescription().id(0)}, CreatureDescription().id(1));

    // Perform action
    data.assignNewIds();

    // Check result
    {
        std::unordered_set<uint64_t> ids;
        for (auto const& creature : data._creatures) {
            ids.insert(creature._id);
        }
        EXPECT_EQ(2, ids.size());
    }
    {
        std::unordered_set<uint64_t> ids;
        for (auto const& object : data._objects) { ids.insert(object._id); }
        ASSERT_EQ(4, ids.size());
    }
}

TEST_F(DescriptionEditTests, assignNewIds_sameCreatureIds)
{
    // Create test data
    auto data = Description()
                    .addCreature({ObjectDescription().id(0), ObjectDescription().id(0)}, CreatureDescription().id(0))
                    .addCreature({ObjectDescription().id(0), ObjectDescription().id(0)}, CreatureDescription().id(0));

    // Perform action
    EXPECT_THROW(data.assignNewIds(), std::runtime_error);
}

TEST_F(DescriptionEditTests, assignNewIds_creatureWithAncestorId_contained)
{
    // Create test data
    auto data = Description()
                    .addCreature({ObjectDescription()}, CreatureDescription().id(2))
                    .addCreature({ObjectDescription()}, CreatureDescription().id(3).ancestorId(2));

    // Perform action
    data.assignNewIds();

    // Check result
    std::unordered_set<uint64_t> ids;
    for (auto const& creature : data._creatures) {
        ids.insert(creature._id);
    }
    EXPECT_EQ(2, ids.size());

    std::optional<CreatureDescription> offspring, ancestor;
    for (auto const& creature : data._creatures) {
        if (creature._ancestorId.has_value()) {
            offspring = creature;
        } else {
            ancestor = creature;
        }
    }
    ASSERT_TRUE(offspring.has_value());
    ASSERT_TRUE(ancestor.has_value());

    EXPECT_EQ(ancestor->_id, offspring->_ancestorId.value());
}

TEST_F(DescriptionEditTests, assignNewIds_creatureWithAncestorId_notContained)
{
    // Create test data
    auto data = Description()
                    .addCreature({ObjectDescription()}, CreatureDescription().id(2))
                    .addCreature({ObjectDescription()}, CreatureDescription().id(3).ancestorId(1));

    // Perform action
    data.assignNewIds();

    // Check result
    std::unordered_set<uint64_t> ids;
    for (auto const& creature : data._creatures) {
        ids.insert(creature._id);
    }
    EXPECT_EQ(2, ids.size());

    std::optional<CreatureDescription> offspring, ancestor;
    for (auto const& creature : data._creatures) {
        if (creature._ancestorId.has_value()) {
            offspring = creature;
        } else {
            ancestor = creature;
        }
    }
    ASSERT_TRUE(offspring.has_value());
    ASSERT_TRUE(ancestor.has_value());

    EXPECT_EQ(1, offspring->_ancestorId.value());
}

TEST_F(DescriptionEditTests, adaptMaxIds)
{
    auto data = Description()
                    .addCreature({ObjectDescription().id(5)}, CreatureDescription().id(3))
                    .addCreature({ObjectDescription()}, CreatureDescription())
                    .energies({
                        EnergyDescription().id(7),
                        EnergyDescription(),
                    });

    EXPECT_LT(data._creatures.at(0)._id, data._creatures.at(1)._id);
    auto cells0 = data.getObjectsForCreature(data._creatures.at(0)._id);
    auto cells1 = data.getObjectsForCreature(data._creatures.at(1)._id);
    EXPECT_LT(cells0.at(0)._id, cells1.at(0)._id);
    EXPECT_LT(data._energies.at(0)._id, data._energies.at(1)._id);
}


TEST_F(DescriptionEditTests, flattenTopology_longDiagonalCreature_lowerRight)
{
    auto const& WorldWidth = 346;
    auto const& WorldHeight = 100;

    std::vector<ObjectDescription> cells;
    for (int i = 0; i < 1000; ++i) {
        cells.emplace_back(ObjectDescription().id(i).pos({toFloat((50 + i) % WorldWidth), toFloat((50 + i) % WorldHeight)}));
    }
    auto data = Description().addCreature(cells, CreatureDescription());
    for (int i = 1; i < 1000; ++i) {
        data.addConnection(i - 1, i);
    }

    DescriptionEditService::get().flattenTopology(data, IntVector2D{WorldWidth, WorldHeight});

    ASSERT_EQ(1, data._creatures.size());

    auto creatureAfter = data._creatures.front();
    auto creatureCells = data.getObjectsForCreature(creatureAfter._id);
    ASSERT_EQ(1000, creatureCells.size());

    for (int i = 0; i < 1000; ++i) {
        auto const& refCell = data.getObjectRef(0);
        auto const& object = data.getObjectRef(i);
        EXPECT_TRUE(approxCompare(toFloat(i), object._pos.x - refCell._pos.x));
        EXPECT_TRUE(approxCompare(toFloat(i), object._pos.y - refCell._pos.y));
    }
}

TEST_F(DescriptionEditTests, flattenTopology_longDiagonalCreature_upperLeft)
{
    auto const& WorldWidth = 346;
    auto const& WorldHeight = 100;

    std::vector<ObjectDescription> cells;
    for (int i = 0; i < 1000; ++i) {
        cells.emplace_back(ObjectDescription().id(i).pos({toFloat((50 - i + WorldWidth) % WorldWidth), toFloat((50 - i + WorldHeight) % WorldHeight)}));
    }
    auto data = Description().addCreature(cells, CreatureDescription());
    for (int i = 1; i < 1000; ++i) {
        data.addConnection(i - 1, i);
    }

    DescriptionEditService::get().flattenTopology(data, IntVector2D{WorldWidth, WorldHeight});

    ASSERT_EQ(1, data._creatures.size());

    auto creatureAfter = data._creatures.front();
    auto creatureCells = data.getObjectsForCreature(creatureAfter._id);
    ASSERT_EQ(1000, creatureCells.size());

    for (int i = 0; i < 1000; ++i) {
        auto const& refCell = data.getObjectRef(0);
        auto const& object = data.getObjectRef(i);
        EXPECT_TRUE(approxCompare(toFloat(i), refCell._pos.x - object._pos.x));
        EXPECT_TRUE(approxCompare(toFloat(i), refCell._pos.y - object._pos.y));
    }
}
