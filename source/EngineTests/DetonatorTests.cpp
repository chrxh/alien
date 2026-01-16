#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/GenomeDescription.h>
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
    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({10.0f, 10.0f}).cellType(DetonatorDescription().countdown(14)),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = actualData.getObjectRef(1);

    EXPECT_EQ(1, actualData._objects.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_FALSE(actualDetonatorCell._signalState == SignalState_Active);
    EXPECT_EQ(14, std::get<DetonatorDescription>(actualDetonatorCell._cellType)._countdown);
    EXPECT_EQ(DetonatorState_Ready, std::get<DetonatorDescription>(actualDetonatorCell._cellType)._state);
}

TEST_F(DetonatorTests, activateDetonator)
{
    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({10.0f, 10.0f}).cellType(DetonatorDescription().countdown(10)),
        ObjectDescription().id(2).pos({11.0f, 10.0f}).cellType(GeneratorDescription()).signalAndState({1, 0, 0, 0, 0, 0, 0, 0}),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = actualData.getObjectRef(1);

    EXPECT_EQ(2, actualData._objects.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(1.0f, actualDetonatorCell._signal._channels[0]));
    EXPECT_EQ(9, std::get<DetonatorDescription>(actualDetonatorCell._cellType)._countdown);
    EXPECT_EQ(DetonatorState_Activated, std::get<DetonatorDescription>(actualDetonatorCell._cellType)._state);
}

TEST_F(DetonatorTests, explosion)
{
    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({10.0f, 10.0f}).cellType(DetonatorDescription().state(DetonatorState_Activated).countdown(10)),
        ObjectDescription().id(2).pos({12.0f, 10.0f}),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(11);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = actualData.getObjectRef(1);
    auto actualOtherCell = actualData.getObjectRef(2);

    EXPECT_EQ(2, actualData._objects.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_FALSE(actualDetonatorCell._signalState == SignalState_Active);
    EXPECT_EQ(0, std::get<DetonatorDescription>(actualDetonatorCell._cellType)._countdown);
    EXPECT_EQ(DetonatorState_Exploded, std::get<DetonatorDescription>(actualDetonatorCell._cellType)._state);
    EXPECT_TRUE(Math::length(actualOtherCell._vel) > NEAR_ZERO);
}

TEST_F(DetonatorTests, chainExplosion)
{
    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({10.0f, 10.0f}).cellType(DetonatorDescription().state(DetonatorState_Activated).countdown(10)),
        ObjectDescription().id(2).pos({12.0f, 10.0f}).cellType(DetonatorDescription().state(DetonatorState_Ready).countdown(10)),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(11);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = actualData.getObjectRef(1);
    auto actualOtherCell = actualData.getObjectRef(2);

    EXPECT_EQ(DetonatorState_Exploded, std::get<DetonatorDescription>(actualDetonatorCell._cellType)._state);
    EXPECT_EQ(DetonatorState_Activated, std::get<DetonatorDescription>(actualOtherCell._cellType)._state);
    EXPECT_EQ(1, std::get<DetonatorDescription>(actualOtherCell._cellType)._countdown);
}

TEST_F(DetonatorTests, explosionAlsoIfDying)
{
    Description data;
    data._objects = {
        ObjectDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellState(CellState_Dying)
            .activationTime(100)
            .cellType(DetonatorDescription().state(DetonatorState_Activated).countdown(10)),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(11);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = actualData.getObjectRef(1);

    EXPECT_EQ(1, actualData._objects.size());
    EXPECT_EQ(DetonatorState_Exploded, std::get<DetonatorDescription>(actualDetonatorCell._cellType)._state);
}
