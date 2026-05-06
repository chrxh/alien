#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SimulationFacade.h>

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
    Desc data;
    data.addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).fixed(true).type(CellDesc().cellState(CellState_Ready)),
        ObjectDesc().id(2).pos({11.0f, 10.0f}).fixed(true).type(CellDesc().cellState(CellState_Ready)),
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
    Desc data;
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
    Desc data;
    data.addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellState(CellState_Constructing)),
        ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().cellState(CellState_Activating)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_EQ(CellState_Activating, actualData.getObjectRef(1).getCellRef()._cellState);
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, noDyingForFixedCells)
{
    auto data = Desc().addCreature({ObjectDesc().id(1).fixed(true).pos({10.0f, 10.0f})});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, cellDiesWhenLastUpdateExceedsInterval)
{
    auto data = Desc().addCreature({ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().lastUpdate(2 * CELL_UPDATE_INTERVAL + 2))});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Dying, actualData.getObjectRef(1).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, cellStaysAliveWhenLastUpdateBelowInterval)
{
    auto data = Desc().addCreature({ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().lastUpdate(0))});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, fixedCellDoesNotDieFromLastUpdate)
{
    auto data = Desc().addCreature({ObjectDesc().id(1).fixed(true).pos({10.0f, 10.0f}).type(CellDesc().lastUpdate(2 * CELL_UPDATE_INTERVAL + 2))});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, isolatedConstructingNonHeadCellDies)
{
    auto data = Desc().addCreature({ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellState(CellState_Constructing).headCell(false))});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2 * CELL_UPDATE_INTERVAL + 1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Dying, actualData.getObjectRef(1).getCellRef()._cellState);
}

TEST_F(CellStateTransitionTests, headCellDoesNotIncrementLastUpdate)
{
    auto data = Desc().addCreature({ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().headCell(true).lastUpdate(0))});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(0, actualData.getObjectRef(1).getCellRef()._lastUpdate);
}

class CellStateTransitionTests_AllStates
    : public CellStateTransitionTests
    , public testing::WithParamInterface<CellState>
{};


INSTANTIATE_TEST_SUITE_P(
    CellStateTransitionTests_AllStates,
    CellStateTransitionTests_AllStates,
    ::testing::Values(CellState_Ready, CellState_Constructing, CellState_Activating, CellState_Dying));

TEST_P(CellStateTransitionTests_AllStates, solid_cell)
{
    auto cellState = GetParam();

    auto data = Desc()
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

    if (cellState == CellState_Activating) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else {
        EXPECT_EQ(cellState, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}

TEST_P(CellStateTransitionTests_AllStates, freeCell_cell)
{
    auto cellState = GetParam();

    auto data = Desc()
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

    if (cellState == CellState_Activating) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else {
        EXPECT_EQ(cellState, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}
