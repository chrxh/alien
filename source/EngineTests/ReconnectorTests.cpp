#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class ReconnectorTests : public IntegrationTestFramework
{
public:
    ReconnectorTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
            _parameters.reconnectorRadius.value[i] = 3.5f;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~ReconnectorTests() = default;

protected:
    // Helper to create a reconnector creature with a trigger signal cell
    Description createReconnectorWithSignal(
        RealVector2D const& pos,
        ReconnectorModeDescription const& mode = ReconnectCreatureDescription(),
        float signalValue = 1.0f,
        int color = 0,
        int lineageId = 0)
    {
        auto reconnectorCell =
            CellDescription().id(1).pos(pos).color(color).cellType(ReconnectorDescription().mode(mode)).signalAndState({signalValue, 0, 0, 0, 0, 0, 0, 0});

        auto data = Description().addCreature(CreatureDescription().id(1).lineageId(lineageId).cells({
            reconnectorCell,
            CellDescription().id(2).pos({pos.x + 1.0f, pos.y}).color(color),
        }));
        data.addConnection(1, 2);
        return data;
    }
};

//*******************************************
//* Structure mode tests
//*******************************************

TEST_F(ReconnectorTests, structureMode_connectToStructure)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectStructureDescription(), 1.0f);

    // Add structure cell within range
    data._cells.emplace_back(CellDescription().id(10).pos({99.0f, 100.0f}).cellType(StructureCellDescription()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getCellRef(1);

    EXPECT_TRUE(actualData.hasConnection(1, 10));
    EXPECT_TRUE(actualReconnector._signal._channels[Channels::ReconnectorSuccess] > NEAR_ZERO);
}

TEST_F(ReconnectorTests, structureMode_ignoreNonStructure)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectStructureDescription(), 1.0f);

    // Add base cell (non-structure) within range
    data._cells.emplace_back(CellDescription().id(10).pos({99.0f, 100.0f}).cellType(BaseDescription()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getCellRef(1);

    EXPECT_FALSE(actualData.hasConnection(1, 10));
    EXPECT_TRUE(approxCompare(0.0f, actualReconnector._signal._channels[Channels::ReconnectorSuccess]));
}

TEST_F(ReconnectorTests, structureMode_outOfRange)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectStructureDescription(), 1.0f);

    // Add structure cell outside range
    data._cells.emplace_back(CellDescription().id(10).pos({95.0f, 100.0f}).cellType(StructureCellDescription()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getCellRef(1);

    EXPECT_FALSE(actualData.hasConnection(1, 10));
    EXPECT_TRUE(approxCompare(0.0f, actualReconnector._signal._channels[Channels::ReconnectorSuccess]));
}

//*******************************************
//* Free cell mode tests
//*******************************************

TEST_F(ReconnectorTests, freeCellMode_connectToFreeCell)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectFreeCellDescription(), 1.0f);

    // Add free cell within range
    data._cells.emplace_back(CellDescription().id(10).pos({99.0f, 100.0f}).cellType(FreeCellDescription()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getCellRef(1);

    EXPECT_TRUE(actualData.hasConnection(1, 10));
    EXPECT_TRUE(actualReconnector._signal._channels[Channels::ReconnectorSuccess] > NEAR_ZERO);
}

TEST_F(ReconnectorTests, freeCellMode_ignoreNonFreeCell)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectFreeCellDescription(), 1.0f);

    // Add base cell (non-free) within range
    data._cells.emplace_back(CellDescription().id(10).pos({99.0f, 100.0f}).cellType(BaseDescription()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getCellRef(1);

    EXPECT_FALSE(actualData.hasConnection(1, 10));
    EXPECT_TRUE(approxCompare(0.0f, actualReconnector._signal._channels[Channels::ReconnectorSuccess]));
}

TEST_F(ReconnectorTests, freeCellMode_colorRestriction_success)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectFreeCellDescription().restrictToColor(1), 1.0f);

    // Add free cell with matching color
    data._cells.emplace_back(CellDescription().id(10).pos({99.0f, 100.0f}).cellType(FreeCellDescription()).color(1));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, freeCellMode_colorRestriction_failed)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectFreeCellDescription().restrictToColor(1), 1.0f);

    // Add free cell with non-matching color
    data._cells.emplace_back(CellDescription().id(10).pos({99.0f, 100.0f}).cellType(FreeCellDescription()).color(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

//*******************************************
//* Creature mode tests
//*******************************************

TEST_F(ReconnectorTests, creatureMode_connectToDifferentCreature)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectCreatureDescription(), 1.0f);

    // Add another creature nearby
    data.addCreature(CreatureDescription().id(2).cells({
        CellDescription().id(10).pos({99.0f, 100.0f}),
        CellDescription().id(11).pos({98.0f, 100.0f}),
    }));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getCellRef(1);

    EXPECT_TRUE(actualData.hasConnection(1, 10));
    EXPECT_TRUE(actualReconnector._signal._channels[Channels::ReconnectorSuccess] > NEAR_ZERO);
}

TEST_F(ReconnectorTests, creatureMode_ignoreOwnCreature)
{
    auto reconnectorCell = CellDescription()
                               .id(1)
                               .pos({100.0f, 100.0f})
                               .cellType(ReconnectorDescription().mode(ReconnectCreatureDescription()))
                               .signalAndState({1, 0, 0, 0, 0, 0, 0, 0});

    // Create a creature with reconnector and potential target in same creature
    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        reconnectorCell,
        CellDescription().id(2).pos({101.0f, 100.0f}),
        CellDescription().id(3).pos({99.0f, 100.0f}),  // Potential target in same creature but not connected to reconnector
    }));
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getCellRef(1);

    // Should not connect to cell in same creature
    EXPECT_FALSE(actualData.hasConnection(1, 3));
    EXPECT_TRUE(approxCompare(0.0f, actualReconnector._signal._channels[Channels::ReconnectorSuccess]));
}

TEST_F(ReconnectorTests, creatureMode_ignoreFreeCells)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectCreatureDescription(), 1.0f);

    // Add free cell within range (not part of a creature)
    data._cells.emplace_back(CellDescription().id(10).pos({99.0f, 100.0f}).cellType(FreeCellDescription()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    // Should not connect to free cells
    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_colorRestriction_success)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectCreatureDescription().restrictToColor(1), 1.0f);

    // Add creature with matching color
    data.addCreature(CreatureDescription().id(2).cells({
        CellDescription().id(10).pos({99.0f, 100.0f}).color(1),
        CellDescription().id(11).pos({98.0f, 100.0f}).color(1),
    }));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_colorRestriction_failed)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectCreatureDescription().restrictToColor(1), 1.0f);

    // Add creature with non-matching color
    data.addCreature(CreatureDescription().id(2).cells({
        CellDescription().id(10).pos({99.0f, 100.0f}).color(0),
        CellDescription().id(11).pos({98.0f, 100.0f}).color(0),
    }));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_minNumCells_success)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectCreatureDescription().minNumCells(2), 1.0f);

    // Add creature with enough cells (numCells >= 2)
    data.addCreature(CreatureDescription().id(2).numCells(3).cells({
        CellDescription().id(10).pos({99.0f, 100.0f}),
        CellDescription().id(11).pos({98.0f, 100.0f}),
    }));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_minNumCells_failed)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectCreatureDescription().minNumCells(5), 1.0f);

    // Add creature with not enough cells (numCells < 5)
    data.addCreature(CreatureDescription().id(2).numCells(3).cells({
        CellDescription().id(10).pos({99.0f, 100.0f}),
        CellDescription().id(11).pos({98.0f, 100.0f}),
    }));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_maxNumCells_success)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectCreatureDescription().maxNumCells(10), 1.0f);

    // Add creature with few enough cells (numCells <= 10)
    data.addCreature(CreatureDescription().id(2).numCells(5).cells({
        CellDescription().id(10).pos({99.0f, 100.0f}),
        CellDescription().id(11).pos({98.0f, 100.0f}),
    }));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_maxNumCells_failed)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectCreatureDescription().maxNumCells(3), 1.0f);

    // Add creature with too many cells (numCells > 3)
    data.addCreature(CreatureDescription().id(2).numCells(10).cells({
        CellDescription().id(10).pos({99.0f, 100.0f}),
        CellDescription().id(11).pos({98.0f, 100.0f}),
    }));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_sameLineage_success)
{
    auto data = createReconnectorWithSignal(
        {100.0f, 100.0f}, ReconnectCreatureDescription().restrictToLineage(LineageRestriction_SameLineage), 1.0f, 0, 5);

    // Add creature with same lineage
    data.addCreature(CreatureDescription().id(2).lineageId(5).cells({
        CellDescription().id(10).pos({99.0f, 100.0f}),
        CellDescription().id(11).pos({98.0f, 100.0f}),
    }));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_sameLineage_failed)
{
    auto data = createReconnectorWithSignal(
        {100.0f, 100.0f}, ReconnectCreatureDescription().restrictToLineage(LineageRestriction_SameLineage), 1.0f, 0, 5);

    // Add creature with different lineage
    data.addCreature(CreatureDescription().id(2).lineageId(6).cells({
        CellDescription().id(10).pos({99.0f, 100.0f}),
        CellDescription().id(11).pos({98.0f, 100.0f}),
    }));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_otherLineage_success)
{
    auto data = createReconnectorWithSignal(
        {100.0f, 100.0f}, ReconnectCreatureDescription().restrictToLineage(LineageRestriction_OtherLineage), 1.0f, 0, 5);

    // Add creature with different lineage
    data.addCreature(CreatureDescription().id(2).lineageId(6).cells({
        CellDescription().id(10).pos({99.0f, 100.0f}),
        CellDescription().id(11).pos({98.0f, 100.0f}),
    }));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_otherLineage_failed)
{
    auto data = createReconnectorWithSignal(
        {100.0f, 100.0f}, ReconnectCreatureDescription().restrictToLineage(LineageRestriction_OtherLineage), 1.0f, 0, 5);

    // Add creature with same lineage
    data.addCreature(CreatureDescription().id(2).lineageId(5).cells({
        CellDescription().id(10).pos({99.0f, 100.0f}),
        CellDescription().id(11).pos({98.0f, 100.0f}),
    }));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

//*******************************************
//* Remove connections tests
//*******************************************

TEST_F(ReconnectorTests, removeConnections_removeStructureConnection)
{
    auto reconnectorCell = CellDescription()
                               .id(1)
                               .pos({100.0f, 100.0f})
                               .cellType(ReconnectorDescription().mode(ReconnectCreatureDescription()))
                               .signalAndState({-1, 0, 0, 0, 0, 0, 0, 0});

    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        reconnectorCell,
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addConnection(1, 2);

    // Add structure cell and connect it to reconnector
    data._cells.emplace_back(CellDescription().id(10).pos({99.0f, 100.0f}).cellType(StructureCellDescription()));
    data.addConnection(1, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getCellRef(1);

    // Connection to structure cell should be removed
    EXPECT_FALSE(actualData.hasConnection(1, 10));
    // Connection to own creature should remain
    EXPECT_TRUE(actualData.hasConnection(1, 2));
    EXPECT_TRUE(actualReconnector._signal._channels[Channels::ReconnectorSuccess] > NEAR_ZERO);
}

TEST_F(ReconnectorTests, removeConnections_removeFreeCellConnection)
{
    auto reconnectorCell = CellDescription()
                               .id(1)
                               .pos({100.0f, 100.0f})
                               .cellType(ReconnectorDescription().mode(ReconnectCreatureDescription()))
                               .signalAndState({-1, 0, 0, 0, 0, 0, 0, 0});

    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        reconnectorCell,
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addConnection(1, 2);

    // Add free cell and connect it to reconnector
    data._cells.emplace_back(CellDescription().id(10).pos({99.0f, 100.0f}).cellType(FreeCellDescription()));
    data.addConnection(1, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getCellRef(1);

    // Connection to free cell should be removed
    EXPECT_FALSE(actualData.hasConnection(1, 10));
    // Connection to own creature should remain
    EXPECT_TRUE(actualData.hasConnection(1, 2));
    EXPECT_TRUE(actualReconnector._signal._channels[Channels::ReconnectorSuccess] > NEAR_ZERO);
}

TEST_F(ReconnectorTests, removeConnections_removeDifferentCreatureConnection)
{
    auto reconnectorCell = CellDescription()
                               .id(1)
                               .pos({100.0f, 100.0f})
                               .cellType(ReconnectorDescription().mode(ReconnectCreatureDescription()))
                               .signalAndState({-1, 0, 0, 0, 0, 0, 0, 0});

    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        reconnectorCell,
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addConnection(1, 2);

    // Add another creature and connect it to reconnector
    data.addCreature(CreatureDescription().id(2).cells({
        CellDescription().id(10).pos({99.0f, 100.0f}),
        CellDescription().id(11).pos({98.0f, 100.0f}),
    }));
    data.addConnection(10, 11);
    data.addConnection(1, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getCellRef(1);

    // Connection to other creature should be removed
    EXPECT_FALSE(actualData.hasConnection(1, 10));
    // Connection to own creature should remain
    EXPECT_TRUE(actualData.hasConnection(1, 2));
    EXPECT_TRUE(actualReconnector._signal._channels[Channels::ReconnectorSuccess] > NEAR_ZERO);
}

TEST_F(ReconnectorTests, removeConnections_keepOwnCreatureConnection)
{
    auto reconnectorCell = CellDescription()
                               .id(1)
                               .pos({100.0f, 100.0f})
                               .cellType(ReconnectorDescription().mode(ReconnectCreatureDescription()))
                               .signalAndState({-1, 0, 0, 0, 0, 0, 0, 0});

    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        reconnectorCell,
        CellDescription().id(2).pos({101.0f, 100.0f}),
        CellDescription().id(3).pos({99.0f, 100.0f}),
    }));
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getCellRef(1);

    // All connections to own creature should remain
    EXPECT_TRUE(actualData.hasConnection(1, 2));
    EXPECT_TRUE(actualData.hasConnection(1, 3));
    // No removal occurred
    EXPECT_TRUE(approxCompare(0.0f, actualReconnector._signal._channels[Channels::ReconnectorSuccess]));
}

//*******************************************
//* General behavior tests
//*******************************************

TEST_F(ReconnectorTests, noTrigger_noAction)
{
    // Create reconnector without active signal
    auto reconnectorCell = CellDescription().id(1).pos({100.0f, 100.0f}).cellType(ReconnectorDescription().mode(ReconnectStructureDescription()));

    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        reconnectorCell,
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addConnection(1, 2);

    // Add structure cell within range
    data._cells.emplace_back(CellDescription().id(10).pos({99.0f, 100.0f}).cellType(StructureCellDescription()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    // No connection should be created without trigger
    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, connectsToClosest)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectStructureDescription(), 1.0f);

    // Add two structure cells, one closer than the other
    data._cells.emplace_back(CellDescription().id(10).pos({98.0f, 100.0f}).cellType(StructureCellDescription()));
    data._cells.emplace_back(CellDescription().id(11).pos({99.0f, 100.0f}).cellType(StructureCellDescription()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    // Should connect to the closer cell (id 11)
    EXPECT_TRUE(actualData.hasConnection(1, 11));
    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, skipAlreadyConnected)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectStructureDescription(), 1.0f);

    // Add structure cell within range and already connected
    data._cells.emplace_back(CellDescription().id(10).pos({99.0f, 100.0f}).cellType(StructureCellDescription()));
    data.addConnection(1, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getCellRef(1);

    // Connection should still exist but no new connection created
    EXPECT_TRUE(actualData.hasConnection(1, 10));
    // Success should be 0 because no new connection was made
    EXPECT_TRUE(approxCompare(0.0f, actualReconnector._signal._channels[Channels::ReconnectorSuccess]));
}

TEST_F(ReconnectorTests, energyConservation)
{
    auto data = createReconnectorWithSignal({100.0f, 100.0f}, ReconnectStructureDescription(), 1.0f);
    data._cells.emplace_back(CellDescription().id(10).pos({99.0f, 100.0f}).cellType(StructureCellDescription()));

    auto originalEnergy = getEnergy(data);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(originalEnergy, getEnergy(actualData)));
}
