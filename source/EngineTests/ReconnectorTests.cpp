#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/GenomeDesc.h>
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
    Desc createReconnectorWithPositiveSignal(
        RealVector2D const& pos,
        ReconnectorModeDesc const& mode = ReconnectCreatureDesc(),
        int color = 0,
        int lineageId = 0)
    {
        auto data = Desc().addCreature({
            ObjectDesc().id(1).pos(pos).color(color).type(CellDesc().cellType(ReconnectorDesc().mode(mode))),
            ObjectDesc().id(2).pos({pos.x + 1.0f, pos.y}).color(color).type(CellDesc().signal({1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),  // Signal on connected cell will propagate
        }, CreatureDesc(), GenomeDesc().lineageId(lineageId));
        data.addConnection(1, 2);
        return data;
    }

    Desc createReconnectorWithNegativeSignal(
        RealVector2D const& pos,
        ReconnectorModeDesc const& mode = ReconnectCreatureDesc(),
        int color = 0,
        int lineageId = 0)
    {
        auto data = Desc().addCreature({
            ObjectDesc().id(1).pos(pos).color(color).type(CellDesc().cellType(ReconnectorDesc().mode(mode))),
            ObjectDesc().id(2).pos({pos.x + 1.0f, pos.y}).color(color).type(CellDesc().signal({-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),  // Signal on connected cell will propagate
        }, CreatureDesc(), GenomeDesc().lineageId(lineageId));
        data.addConnection(1, 2);
        return data;
    }
};

//*******************************************
//* Structure mode tests
//*******************************************

TEST_F(ReconnectorTests, structureMode_connectToStructure)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectStructureDesc());

    // Add structure cell within range
    data._objects.emplace_back(ObjectDesc().id(10).pos({99.0f, 100.0f}).type(StructureDesc()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getObjectRef(1);

    EXPECT_TRUE(actualData.hasConnection(1, 10));
    EXPECT_TRUE(actualReconnector.getCellRef()._signal._channels[Channels::ReconnectorSuccess] > NEAR_ZERO);
}

TEST_F(ReconnectorTests, structureMode_ignoreNonStructure)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectStructureDesc());

    // Add free cell (non-structure) within range
    data._objects.emplace_back(ObjectDesc().id(10).pos({99.0f, 100.0f}).type(FreeCellDesc()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getObjectRef(1);

    EXPECT_FALSE(actualData.hasConnection(1, 10));
    EXPECT_TRUE(approxCompare(0.0f, actualReconnector.getCellRef()._signal._channels[Channels::ReconnectorSuccess]));
}

TEST_F(ReconnectorTests, structureMode_outOfRange)
{
    auto range = _parameters.reconnectorRadius.value[0];
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectStructureDesc());

    // Add structure cell outside range
    data._objects.emplace_back(ObjectDesc().id(10).pos({100.0f - range - 0.1f, 100.0f}).type(StructureDesc()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getObjectRef(1);

    EXPECT_FALSE(actualData.hasConnection(1, 10));
    EXPECT_TRUE(approxCompare(0.0f, actualReconnector.getCellRef()._signal._channels[Channels::ReconnectorSuccess]));
}

//*******************************************
//* Free cell mode tests
//*******************************************

TEST_F(ReconnectorTests, freeCellMode_connectToFreeCell)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectFreeCellDesc());

    // Add free cell within range
    data._objects.emplace_back(ObjectDesc().id(10).pos({99.0f, 100.0f}).type(FreeCellDesc()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getObjectRef(1);

    EXPECT_TRUE(actualData.hasConnection(1, 10));
    EXPECT_TRUE(actualReconnector.getCellRef()._signal._channels[Channels::ReconnectorSuccess] > NEAR_ZERO);
}

TEST_F(ReconnectorTests, freeCellMode_ignoreNonFreeCell)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectFreeCellDesc());

    // Add structure cell (non-free) within range
    data._objects.emplace_back(ObjectDesc().id(10).pos({99.0f, 100.0f}).type(StructureDesc()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getObjectRef(1);

    EXPECT_FALSE(actualData.hasConnection(1, 10));
    EXPECT_TRUE(approxCompare(0.0f, actualReconnector.getCellRef()._signal._channels[Channels::ReconnectorSuccess]));
}

TEST_F(ReconnectorTests, freeCellMode_colorRestriction_success)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectFreeCellDesc().restrictToColor(1));

    // Add free cell with matching color
    data._objects.emplace_back(ObjectDesc().id(10).pos({99.0f, 100.0f}).color(1).type(FreeCellDesc()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, freeCellMode_colorRestriction_failed)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectFreeCellDesc().restrictToColor(1));

    // Add free cell with non-matching color
    data._objects.emplace_back(ObjectDesc().id(10).pos({99.0f, 100.0f}).color(0).type(FreeCellDesc()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

//*******************************************
//* Creature mode tests
//*******************************************

TEST_F(ReconnectorTests, creatureMode_connectToDifferentCreature)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectCreatureDesc());

    // Add another creature nearby
    data.addCreature({
        ObjectDesc().id(10).pos({99.0f, 100.0f}),
        ObjectDesc().id(11).pos({98.0f, 100.0f}),
    });
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getObjectRef(1);

    EXPECT_TRUE(actualData.hasConnection(1, 10));
    EXPECT_TRUE(actualReconnector.getCellRef()._signal._channels[Channels::ReconnectorSuccess] > NEAR_ZERO);
}

TEST_F(ReconnectorTests, creatureMode_ignoreOwnCreature)
{
    // Create a creature with reconnector, generator, and potential target in same creature
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().neuralNetwork(NeuralNetDesc().bias(0, 1.0f)).cellType(ReconnectorDesc().mode(ReconnectCreatureDesc()))),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
        ObjectDesc().id(3).pos({99.0f, 100.0f}),  // Potential target in same creature but not connected to reconnector
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getObjectRef(1);

    // Should not connect to cell in same creature
    EXPECT_FALSE(actualData.hasConnection(1, 3));
    EXPECT_TRUE(approxCompare(0.0f, actualReconnector.getCellRef()._signal._channels[Channels::ReconnectorSuccess]));
}

TEST_F(ReconnectorTests, creatureMode_ignoreFreeCells)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectCreatureDesc());

    // Add free cell within range (not part of a creature)
    data._objects.emplace_back(ObjectDesc().id(10).pos({99.0f, 100.0f}).type(FreeCellDesc()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    // Should not connect to free cells
    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_colorRestriction_success)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectCreatureDesc().restrictToColor(1));

    // Add creature with matching color
    data.addCreature({
        ObjectDesc().id(10).pos({99.0f, 100.0f}).color(1),
        ObjectDesc().id(11).pos({98.0f, 100.0f}).color(1),
    });
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_colorRestriction_failed)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectCreatureDesc().restrictToColor(1));

    // Add creature with non-matching color
    data.addCreature({
        ObjectDesc().id(10).pos({99.0f, 100.0f}).color(0),
        ObjectDesc().id(11).pos({98.0f, 100.0f}).color(0),
    });
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_minNumCells_success)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectCreatureDesc().minNumCells(2));

    // Add creature with enough cells (numCells >= 2)
    data.addCreature({
        ObjectDesc().id(10).pos({99.0f, 100.0f}),
        ObjectDesc().id(11).pos({98.0f, 100.0f}),
    }, CreatureDesc().numObjects(3));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_minNumCells_failed)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectCreatureDesc().minNumCells(5));

    // Add creature with not enough cells (numCells < 5)
    data.addCreature({
        ObjectDesc().id(10).pos({99.0f, 100.0f}),
        ObjectDesc().id(11).pos({98.0f, 100.0f}),
    }, CreatureDesc().numObjects(3));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_maxNumCells_success)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectCreatureDesc().maxNumCells(10));

    // Add creature with few enough cells (numCells <= 10)
    data.addCreature({
        ObjectDesc().id(10).pos({99.0f, 100.0f}),
        ObjectDesc().id(11).pos({98.0f, 100.0f}),
    }, CreatureDesc().numObjects(5));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_maxNumCells_failed)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectCreatureDesc().maxNumCells(3));

    // Add creature with too many cells (numCells > 3)
    data.addCreature({
        ObjectDesc().id(10).pos({99.0f, 100.0f}),
        ObjectDesc().id(11).pos({98.0f, 100.0f}),
        ObjectDesc().id(12).pos({97.0f, 100.0f}),
        ObjectDesc().id(13).pos({96.0f, 100.0f}),
    });
    data.addConnection(10, 11);
    data.addConnection(11, 12);
    data.addConnection(12, 13);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_sameLineage_success)
{
    auto data = createReconnectorWithPositiveSignal(
        {100.0f, 100.0f}, ReconnectCreatureDesc().restrictToLineage(LineageRestriction_SameLineage), 0, 5);

    // Add creature with same lineage
    data.addCreature({
        ObjectDesc().id(10).pos({99.0f, 100.0f}),
        ObjectDesc().id(11).pos({98.0f, 100.0f}),
    }, CreatureDesc(), GenomeDesc().lineageId(5));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_sameLineage_failed)
{
    auto data = createReconnectorWithPositiveSignal(
        {100.0f, 100.0f}, ReconnectCreatureDesc().restrictToLineage(LineageRestriction_SameLineage), 0, 5);

    // Add creature with different lineage
    data.addCreature({
        ObjectDesc().id(10).pos({99.0f, 100.0f}),
        ObjectDesc().id(11).pos({98.0f, 100.0f}),
    }, CreatureDesc(), GenomeDesc().lineageId(6));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_otherLineage_success)
{
    auto data = createReconnectorWithPositiveSignal(
        {100.0f, 100.0f}, ReconnectCreatureDesc().restrictToLineage(LineageRestriction_OtherLineage), 0, 5);

    // Add creature with different lineage
    data.addCreature({
        ObjectDesc().id(10).pos({99.0f, 100.0f}),
        ObjectDesc().id(11).pos({98.0f, 100.0f}),
    }, CreatureDesc(), GenomeDesc().lineageId(6));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, creatureMode_otherLineage_failed)
{
    auto data = createReconnectorWithPositiveSignal(
        {100.0f, 100.0f}, ReconnectCreatureDesc().restrictToLineage(LineageRestriction_OtherLineage), 0, 5);

    // Add creature with same lineage
    data.addCreature({
        ObjectDesc().id(10).pos({99.0f, 100.0f}),
        ObjectDesc().id(11).pos({98.0f, 100.0f}),
    }, CreatureDesc(), GenomeDesc().lineageId(5));
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

//*******************************************
//* Remove connections tests
//*******************************************

TEST_F(ReconnectorTests, removeConnections_removeStructureConnection)
{
    auto data = createReconnectorWithNegativeSignal({100.0f, 100.0f}, ReconnectCreatureDesc());

    // Add structure cell and connect it to reconnector
    data._objects.emplace_back(ObjectDesc().id(10).pos({99.0f, 100.0f}).type(StructureDesc()));
    data.addConnection(1, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getObjectRef(1);

    // Connection to structure cell should be removed
    EXPECT_FALSE(actualData.hasConnection(1, 10));
    // Connection to own creature should remain
    EXPECT_TRUE(actualData.hasConnection(1, 2));
    EXPECT_TRUE(actualReconnector.getCellRef()._signal._channels[Channels::ReconnectorSuccess] > NEAR_ZERO);
}

TEST_F(ReconnectorTests, removeConnections_removeFreeObjectConnection)
{
    auto data = createReconnectorWithNegativeSignal({100.0f, 100.0f}, ReconnectCreatureDesc());

    // Add free cell and connect it to reconnector
    data._objects.emplace_back(ObjectDesc().id(10).pos({99.0f, 100.0f}).type(FreeCellDesc()));
    data.addConnection(1, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getObjectRef(1);

    // Connection to free cell should be removed
    EXPECT_FALSE(actualData.hasConnection(1, 10));
    // Connection to own creature should remain
    EXPECT_TRUE(actualData.hasConnection(1, 2));
    EXPECT_TRUE(actualReconnector.getCellRef()._signal._channels[Channels::ReconnectorSuccess] > NEAR_ZERO);
}

TEST_F(ReconnectorTests, removeConnections_removeDifferentCreatureConnection)
{
    auto data = createReconnectorWithNegativeSignal({100.0f, 100.0f}, ReconnectCreatureDesc());

    // Add another creature and connect it to reconnector
    data.addCreature({
        ObjectDesc().id(10).pos({99.0f, 100.0f}),
        ObjectDesc().id(11).pos({98.0f, 100.0f}),
    });
    data.addConnection(10, 11);
    data.addConnection(1, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getObjectRef(1);

    // Connection to other creature should be removed
    EXPECT_FALSE(actualData.hasConnection(1, 10));
    // Connection to own creature should remain
    EXPECT_TRUE(actualData.hasConnection(1, 2));
    EXPECT_TRUE(actualReconnector.getCellRef()._signal._channels[Channels::ReconnectorSuccess] > NEAR_ZERO);
}

TEST_F(ReconnectorTests, removeConnections_keepOwnCreatureConnection)
{
    // Create creature with reconnector and additional object, signal on connected cell
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(ReconnectorDesc().mode(ReconnectCreatureDesc()))),
        ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().signal({-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
        ObjectDesc().id(3).pos({99.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getObjectRef(1);

    // All connections to own creature should remain
    EXPECT_TRUE(actualData.hasConnection(1, 2));
    EXPECT_TRUE(actualData.hasConnection(1, 3));
    // No removal occurred
    EXPECT_TRUE(approxCompare(0.0f, actualReconnector.getCellRef()._signal._channels[Channels::ReconnectorSuccess]));
}

//*******************************************
//* General behavior tests
//*******************************************

TEST_F(ReconnectorTests, noTrigger_noAction)
{
    // Create reconnector without active signal (no generator)
    auto reconnectorCell = ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(ReconnectorDesc().mode(ReconnectStructureDesc())));

    auto data = Desc().addCreature({
        reconnectorCell,
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add structure cell within range
    data._objects.emplace_back(ObjectDesc().id(10).pos({99.0f, 100.0f}).type(StructureDesc()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Run several timesteps without trigger

    auto actualData = _simulationFacade->getSimulationData();

    // No connection should be created without trigger
    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, connectsToClosest)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectStructureDesc());

    // Add two structure cells, one closer than the other
    data._objects.emplace_back(ObjectDesc().id(10).pos({98.0f, 100.0f}).type(StructureDesc()));
    data._objects.emplace_back(ObjectDesc().id(11).pos({99.0f, 100.0f}).type(StructureDesc()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    // Should connect to the closer cell (id 11)
    EXPECT_TRUE(actualData.hasConnection(1, 11));
    EXPECT_FALSE(actualData.hasConnection(1, 10));
}

TEST_F(ReconnectorTests, skipAlreadyConnected)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectStructureDesc());

    // Add structure cell within range and already connected
    data._objects.emplace_back(ObjectDesc().id(10).pos({99.0f, 100.0f}).type(StructureDesc()));
    data.addConnection(1, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getObjectRef(1);

    // Connection should still exist but no new connection created
    EXPECT_TRUE(actualData.hasConnection(1, 10));
    // Success should be 0 because no new connection was made
    EXPECT_TRUE(approxCompare(0.0f, actualReconnector.getCellRef()._signal._channels[Channels::ReconnectorSuccess]));
}

TEST_F(ReconnectorTests, energyConservation)
{
    auto data = createReconnectorWithPositiveSignal({100.0f, 100.0f}, ReconnectStructureDesc());
    data._objects.emplace_back(ObjectDesc().id(10).pos({99.0f, 100.0f}).type(StructureDesc()));

    auto originalEnergy = getEnergy(data);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(originalEnergy, getEnergy(actualData)));
}

TEST_F(ReconnectorTests, rayNotBlockedByDifferentCreatureConnections)
{
    auto range = 4.0f;
    _parameters.reconnectorRadius.value[0] = range;
    _simulationFacade->setSimulationParameters(_parameters);

    // Create attacker with connections that block the attack ray
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(ReconnectorDesc().mode(ReconnectStructureDesc()))),
        ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().signal({-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
        // Create a connection that crosses the ray path to target at (100, 99)
        ObjectDesc().id(3).pos({99.0f, 99.0f}),
        ObjectDesc().id(4).pos({101.0f, 99.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(3, 4);
    data.addConnection(1, 4);

    // Add target creature below (ray to target is blocked by connection 3-4)
    data._objects.emplace_back(ObjectDesc().id(10).pos({100.0f, 100.0f - (range - 1.0f)}).type(StructureDesc()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnector = actualData.getObjectRef(1);

    EXPECT_FALSE(actualData.hasConnection(1, 10));
    EXPECT_TRUE(approxCompare(0.0f, actualReconnector.getCellRef()._signal._channels[Channels::ReconnectorSuccess]));
}
