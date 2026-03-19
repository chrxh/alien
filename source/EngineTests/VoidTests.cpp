#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/GenomeDesc.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class VoidTests : public IntegrationTestFramework
{
public:
    VoidTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~VoidTests() = default;
};

TEST_F(VoidTests, doNothing_underConstruction)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellType(VoidDesc()).cellState(CellState_Constructing)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(1, actualData._objects.size());
}

TEST_F(VoidTests, doNothing_activating)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellType(VoidDesc()).cellState(CellState_Activating).activationTime(10)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(1, actualData._objects.size());
}

TEST_F(VoidTests, destroyWhenReady_singleCell)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellType(VoidDesc()).cellState(CellState_Ready)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(0, actualData._objects.size());
}

enum class VoidSimulationMode
{
    Normal,
    Preview,
    NonDetailPreview
};

class VoidTests_SimulationMode
    : public VoidTests
    , public testing::WithParamInterface<VoidSimulationMode>
{
public:
    void runTimesteps(Desc& data, int timesteps)
    {
        auto mode = GetParam();
        if (mode == VoidSimulationMode::Normal) {
            _simulationFacade->setSimulationData(data);
            _simulationFacade->calcTimesteps(timesteps);
        } else {
            _simulationFacade->setPreviewData(data);
            _simulationFacade->calcTimestepsForPreview(timesteps, mode == VoidSimulationMode::Preview);
        }
    }

    Desc getResult()
    {
        auto mode = GetParam();
        if (mode == VoidSimulationMode::Normal) {
            return _simulationFacade->getSimulationData();
        } else {
            return _simulationFacade->getPreviewData();
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    VoidTests_SimulationMode,
    VoidTests_SimulationMode,
    ::testing::Values(VoidSimulationMode::Normal, VoidSimulationMode::Preview, VoidSimulationMode::NonDetailPreview));

TEST_P(VoidTests_SimulationMode, destroyWhenReady_neighborsNotDetaching)
{
    auto data = Desc()
                    .addCreature({
                        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellState(CellState_Ready).usableEnergy(100.0f).cellType(BaseDesc())),
                        ObjectDesc().id(2).pos({10.5f, 10.866f}).type(CellDesc().cellState(CellState_Ready).usableEnergy(200.0f).cellType(VoidDesc())),
                        ObjectDesc().id(3).pos({11.0f, 10.0f}).type(CellDesc().cellState(CellState_Ready).usableEnergy(100.0f).cellType(BaseDesc())),
                    })
                    .addConnection(1, 2)
                    .addConnection(2, 3)
                    .addConnection(1, 3);

    runTimesteps(data, TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = getResult();
    EXPECT_EQ(2, actualData._objects.size());
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(3).getCellRef()._cellState);

    auto totalEnergy = getEnergy(actualData);
    EXPECT_TRUE(approxCompare(getEnergy(data), totalEnergy));

    if (GetParam() == VoidSimulationMode::Normal) {
        _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION * 10);

        actualData = _simulationFacade->getSimulationData();
        EXPECT_EQ(2, actualData._objects.size());
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(3).getCellRef()._cellState);
    }
}
