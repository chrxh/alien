#include <gtest/gtest.h>

#include <ranges>

#include <Base/Interface/Math.h>

#include <Base/Interface/NumberGenerator.h>

#include <Data/Interface/DescEditService.h>
#include <Data/Interface/Descs.h>

#include <Engine/Interface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class CellStateTransitionTests : public IntegrationTestFramework
{
public:
    CellStateTransitionTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.cellDeathProbability.baseValue[i] = 0;
            _parameters.radiationType1_strength.baseValue[i] = 0;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    CellState getFirstHeadCellState()
    {
        auto actualData = _simulationFacade->getSimulationData();
        uint64_t firstHeadCellId = std::numeric_limits<uint64_t>::max();
        for (auto const& object : actualData._objects) {
            if (object.getCellRef()._headCell) {
                firstHeadCellId = std::min(firstHeadCellId, object._id);
            }
        }
        return actualData.getObjectRef(firstHeadCellId).getCellRef()._cellState;
    }

    ObjectTypeDesc getObjectTypeDesc(ObjectType objectType)
    {
        if (objectType == ObjectType_Solid) {
            return SolidDesc();
        } else if (objectType == ObjectType_Fluid) {
            return FluidDesc();
        } else if (objectType == ObjectType_FreeCell) {
            return FreeCellDesc();
        } else {
            return CellDesc().cellType(BaseDesc());
        }
    }
};

TEST_F(CellStateTransitionTests, ready_ready)
{
    ContentDesc data;
    data.addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).isStatic(true).type(CellDesc().cellState(CellState_Ready)),
        ObjectDesc().id(2).pos({11.0f, 10.0f}).isStatic(true).type(CellDesc().cellState(CellState_Ready)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, ready_dying)
{
    ContentDesc data;
    data.addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellState(CellState_Ready)),
        ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().cellState(CellState_Dying)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
    EXPECT_EQ(CellState_Dying, actualData.getObjectRef(2).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, underConstruction_activating)
{
    ContentDesc data;
    data.addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellState(CellState_UnderConstruction)),
        ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().cellState(CellState_BeingActivated)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_EQ(CellState_BeingActivated, actualData.getObjectRef(1).getCellRef()._cellState);
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, noDyingForFixedCells)
{
    auto data = ContentDesc().addCreature({ObjectDesc().id(1).isStatic(true).pos({10.0f, 10.0f})});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, cellDiesWhenLastUpdateExceedsInterval)
{
    auto data = ContentDesc().addCreature({ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().lastUpdate(2 * CELL_UPDATE_INTERVAL + 2))});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Dying, actualData.getObjectRef(1).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, cellStaysAliveWhenLastUpdateBelowInterval)
{
    auto data = ContentDesc().addCreature({ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().lastUpdate(0))});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, fixedCellDoesNotDieFromLastUpdate)
{
    auto data = ContentDesc().addCreature({ObjectDesc().id(1).isStatic(true).pos({10.0f, 10.0f}).type(CellDesc().lastUpdate(2 * CELL_UPDATE_INTERVAL + 2))});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, isolatedUnderConstructionNonHeadCellDies)
{
    auto data = ContentDesc().addCreature({ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellState(CellState_UnderConstruction).headCell(false))});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2 * CELL_UPDATE_INTERVAL + 1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Dying, actualData.getObjectRef(1).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, headCellDoesNotIncrementLastUpdate)
{
    auto data = ContentDesc().addCreature({ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().headCell(true).lastUpdate(0))});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(0, actualData.getObjectRef(1).getCellRef()._lastUpdate);
}

TEST_F(CellStateTransitionTests, headCellDiesWhenHostConstructorDiesDuringConstruction)
{
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .cellState(CellState_InstantDying)
                          .usableEnergy(_parameters.normalCellEnergy.value[0] * 3.5f)
                          .constructor(ConstructorDesc().geneIndex(0).separation(true))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc()})}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(CELL_UPDATE_INTERVAL + 2);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._objects.size());
    auto const& offspringCell = actualData.getOtherObjectRef(1).getCellRef();
    EXPECT_TRUE(offspringCell._headCell);
    EXPECT_EQ(CellState_Dying, offspringCell._cellState);
}

TEST_F(CellStateTransitionTests, headCellDiesWhenUnfinishedCreatureHasNoHost)
{
    auto data = ContentDesc().addCreature({ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellState(CellState_UnderConstruction).headCell(true))});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(CELL_UPDATE_INTERVAL + 2);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_EQ(CellState_Dying, actualData.getObjectRef(1).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, headCellOfAbandonedChainDiesWhenHostStartsNewChain)
{
    _parameters.minCellEnergy.baseValue[1] = _parameters.normalCellEnergy.value[0] * 1.5f;
    _parameters.cellDeathProbability.baseValue[1] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    std::vector<NodeDesc> nodes(25);
    nodes.at(1) = NodeDesc().color(1);
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(_parameters.normalCellEnergy.value[0] * 40.0f)
                          .constructor(ConstructorDesc().autoTriggerInterval(6).geneIndex(0).separation(true).numBranches(1))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().nodes(nodes)}));

    _simulationFacade->setSimulationData(data);

    // The newest cell of the first chain dies, afterwards the host builds an intact chain
    _simulationFacade->calcTimesteps(10);
    _parameters.cellDeathProbability.baseValue[1] = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->calcTimesteps(2 * CELL_UPDATE_INTERVAL + 2 - 10);

    // Both chains have a head cell, the one of the abandoned chain is the older one
    EXPECT_EQ(CellState_Dying, getFirstHeadCellState());
}

TEST_F(CellStateTransitionTests, headCellOfFinishedCreatureDiesWhenActivationDoesNotReachIt)
{
    _parameters.minCellEnergy.baseValue[1] = _parameters.normalCellEnergy.value[0] * 1.5f;
    _parameters.cellDeathProbability.baseValue[1] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = ContentDesc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(_parameters.normalCellEnergy.value[0] * 5.0f)
                          .constructor(ConstructorDesc().autoTriggerInterval(6).geneIndex(0).separation(true).numBranches(1))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc().color(1)})}));

    _simulationFacade->setSimulationData(data);

    // The last cell dies immediately, so that the activation never reaches the head cell
    _simulationFacade->calcTimesteps(CELL_UPDATE_INTERVAL + 2);
    EXPECT_EQ(CellState_UnderConstruction, getFirstHeadCellState());

    _simulationFacade->calcTimesteps(CELL_UPDATE_INTERVAL);
    EXPECT_EQ(CellState_Dying, getFirstHeadCellState());
}

TEST_F(CellStateTransitionTests, headCellOfLongChainSurvivesWhileActivationIsRunning)
{
    auto constexpr ChainLength = 150;

    std::vector<ObjectDesc> cells;
    for (auto i : std::views::iota(0, ChainLength)) {
        auto cellState = i == ChainLength - 1 ? CellState_BeingActivated : CellState_UnderConstruction;
        cells.emplace_back(ObjectDesc().id(i + 1).pos({10.0f + toFloat(i), 10.0f}).type(CellDesc().cellState(cellState).headCell(i == 0)));
    }
    auto data = ContentDesc().addCreature(cells);

    // The first connection of each cell points to the end of the chain, i.e. in the direction of the activation
    for (auto i : std::views::iota(0, ChainLength - 1) | std::views::reverse) {
        data.addConnection(i + 1, i + 2);
    }

    _simulationFacade->setSimulationData(data);

    // The activation needs more steps to reach the head cell than the host would have to confirm the creature
    _simulationFacade->calcTimesteps(CELL_UPDATE_INTERVAL + 2);
    EXPECT_EQ(CellState_UnderConstruction, getFirstHeadCellState());

    _simulationFacade->calcTimesteps(CELL_UPDATE_INTERVAL);
    EXPECT_EQ(CellState_Ready, getFirstHeadCellState());
}

class CellStateTransitionTests_AllStates
    : public CellStateTransitionTests
    , public testing::WithParamInterface<CellState>
{};


INSTANTIATE_TEST_SUITE_P(
    CellStateTransitionTests_AllStates,
    CellStateTransitionTests_AllStates,
    ::testing::Values(CellState_Ready, CellState_UnderConstruction, CellState_BeingActivated, CellState_Dying));

TEST_P(CellStateTransitionTests_AllStates, solid_cell)
{
    auto cellState = GetParam();

    auto data = ContentDesc()
                    .addObjects({
                        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(SolidDesc()),
                    })
                    .addCreature({
                        ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().cellState(cellState)),
                    })
                    .addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (cellState == CellState_BeingActivated) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else {
        EXPECT_EQ(cellState, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}

TEST_P(CellStateTransitionTests_AllStates, freeCell_cell)
{
    auto cellState = GetParam();

    auto data = ContentDesc()
                    .addObjects({
                        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(FreeCellDesc()),
                    })
                    .addCreature({
                        ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().cellState(cellState)),
                    })
                    .addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (cellState == CellState_BeingActivated) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else {
        EXPECT_EQ(cellState, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}
