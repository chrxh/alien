#include <gtest/gtest.h>

#include "Base/Math.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionConverterService.h"
#include "EngineInterface/GenomeDescriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class DetonatorTests : public IntegrationTestFramework
{
public:
    static SimulationParameters getParameters()
    {
        SimulationParameters result;
        result.innerFriction.value = 0;
        result.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.radiationType1_strength.baseValue[i] = 0;
        }
        return result;
    }
    DetonatorTests()
        : IntegrationTestFramework(getParameters())
    {}

    ~DetonatorTests() = default;
};

TEST_F(DetonatorTests, doNothing)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellType(DetonatorDescription().countdown(14)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = getCell(actualData, 1);

    EXPECT_EQ(1, actualData._cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(0.0f, actualDetonatorCell._signal->_channels[0]));
    EXPECT_EQ(14, std::get<DetonatorDescription>(actualDetonatorCell._cellTypeData)._countdown);
    EXPECT_EQ(DetonatorState_Ready, std::get<DetonatorDescription>(actualDetonatorCell._cellTypeData)._state);
}

TEST_F(DetonatorTests, activateDetonator)
{
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({10.0f, 10.0f})
             .cellType(DetonatorDescription().countdown(10)),
         CellDescription()
             .id(2)
             .pos({11.0f, 10.0f})
             .cellType(OscillatorDescription())
             .signal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = getCell(actualData, 1);

    EXPECT_EQ(2, actualData._cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(1.0f, actualDetonatorCell._signal->_channels[0]));
    EXPECT_EQ(9, std::get<DetonatorDescription>(actualDetonatorCell._cellTypeData)._countdown);
    EXPECT_EQ(DetonatorState_Activated, std::get<DetonatorDescription>(actualDetonatorCell._cellTypeData)._state);
}

TEST_F(DetonatorTests, explosion)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellType(DetonatorDescription().state(DetonatorState_Activated).countdown(10)),
        CellDescription().id(2).pos({12.0f, 10.0f}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(6 * 10 + 1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = getCell(actualData, 1);
    auto actualOtherCell = getCell(actualData, 2);

    EXPECT_EQ(2, actualData._cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(0.0f, actualDetonatorCell._signal->_channels[0]));
    EXPECT_EQ(0, std::get<DetonatorDescription>(actualDetonatorCell._cellTypeData)._countdown);
    EXPECT_EQ(DetonatorState_Exploded, std::get<DetonatorDescription>(actualDetonatorCell._cellTypeData)._state);
    EXPECT_TRUE(Math::length(actualOtherCell._vel) > NEAR_ZERO);
}

TEST_F(DetonatorTests, chainExplosion)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(DetonatorDescription().state(DetonatorState_Activated).countdown(10)),
        CellDescription()
            .id(2)
            .pos({12.0f, 10.0f})
            .cellType(DetonatorDescription().state(DetonatorState_Ready).countdown(10)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(6 * 11 + 1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = getCell(actualData, 1);
    auto actualOtherCell = getCell(actualData, 2);

    EXPECT_EQ(DetonatorState_Exploded, std::get<DetonatorDescription>(actualDetonatorCell._cellTypeData)._state);
    EXPECT_EQ(DetonatorState_Activated, std::get<DetonatorDescription>(actualOtherCell._cellTypeData)._state);
    EXPECT_EQ(0, std::get<DetonatorDescription>(actualOtherCell._cellTypeData)._countdown);
}

TEST_F(DetonatorTests, explosionIfDying)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .livingState(LivingState_Dying)
            .activationTime(100)
            .cellType(DetonatorDescription().state(DetonatorState_Activated).countdown(10)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(6 * 10 + 1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDetonatorCell = getCell(actualData, 1);

    EXPECT_EQ(1, actualData._cells.size());
    EXPECT_EQ(DetonatorState_Exploded, std::get<DetonatorDescription>(actualDetonatorCell._cellTypeData)._state);
}
