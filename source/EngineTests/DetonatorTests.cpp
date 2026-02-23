#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/GenomeDesc.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class DetonatorTests : public IntegrationTestFramework
{
public:
    DetonatorTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
            _parameters.detonatorChainExplosionProbability.value[i] = 1.0f;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~DetonatorTests() = default;
};

TEST_F(DetonatorTests, doNothing)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellType(DetonatorDesc().countdown(14))),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = actualData.getObjectRef(1);

    EXPECT_EQ(1, actualData._objects.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(0.0f, actualDetonatorCell.getCellRef()._signal._channels[0]));
    EXPECT_EQ(14, std::get<DetonatorDesc>(actualDetonatorCell.getCellRef()._cellType)._countdown);
    EXPECT_EQ(DetonatorState_Ready, std::get<DetonatorDesc>(actualDetonatorCell.getCellRef()._cellType)._state);
}

TEST_F(DetonatorTests, activateDetonator)
{
    auto data = Desc().addCreature({
        ObjectDesc()
            .id(1)
            .pos({10.0f, 10.0f})
            .type(
                CellDesc().neuralNetwork(NeuralNetDesc().bias(0, 1.0f)).cellType(DetonatorDesc().countdown(10))),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = actualData.getObjectRef(1);

    EXPECT_EQ(1, actualData._objects.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(1.0f, actualDetonatorCell.getCellRef()._signal._channels[0]));
    EXPECT_EQ(9, std::get<DetonatorDesc>(actualDetonatorCell.getCellRef()._cellType)._countdown);
    EXPECT_EQ(DetonatorState_Activated, std::get<DetonatorDesc>(actualDetonatorCell.getCellRef()._cellType)._state);
}

TEST_F(DetonatorTests, explosion)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellType(DetonatorDesc().state(DetonatorState_Activated).countdown(10))),
        ObjectDesc().id(2).pos({12.0f, 10.0f}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(11 * TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = actualData.getObjectRef(1);
    auto actualOtherObject = actualData.getObjectRef(2);

    EXPECT_EQ(2, actualData._objects.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(0.0f, actualDetonatorCell.getCellRef()._signal._channels[0]));
    EXPECT_EQ(0, std::get<DetonatorDesc>(actualDetonatorCell.getCellRef()._cellType)._countdown);
    EXPECT_EQ(DetonatorState_Exploded, std::get<DetonatorDesc>(actualDetonatorCell.getCellRef()._cellType)._state);
    EXPECT_TRUE(Math::length(actualOtherObject._vel) > NEAR_ZERO);
}

TEST_F(DetonatorTests, chainExplosion)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellType(DetonatorDesc().state(DetonatorState_Activated).countdown(10))),
        ObjectDesc().id(2).pos({12.0f, 10.0f}).type(CellDesc().cellType(DetonatorDesc().state(DetonatorState_Ready).countdown(10))),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(11 * TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = actualData.getObjectRef(1);
    auto actualOtherObject = actualData.getObjectRef(2);

    EXPECT_EQ(DetonatorState_Exploded, std::get<DetonatorDesc>(actualDetonatorCell.getCellRef()._cellType)._state);
    EXPECT_EQ(DetonatorState_Activated, std::get<DetonatorDesc>(actualOtherObject.getCellRef()._cellType)._state);
    EXPECT_EQ(1, std::get<DetonatorDesc>(actualOtherObject.getCellRef()._cellType)._countdown);
}

TEST_F(DetonatorTests, explosionAlsoIfDying)
{
    auto data = Desc().addCreature({
        ObjectDesc()
            .id(1)
            .pos({10.0f, 10.0f})
            .type(
                CellDesc().cellState(CellState_Dying).activationTime(100).cellType(DetonatorDesc().state(DetonatorState_Activated).countdown(10))),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(11 * TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = actualData.getObjectRef(1);

    EXPECT_EQ(1, actualData._objects.size());
    EXPECT_EQ(DetonatorState_Exploded, std::get<DetonatorDesc>(actualDetonatorCell.getCellRef()._cellType)._state);
}
