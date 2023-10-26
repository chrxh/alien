#include <gtest/gtest.h>

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionConverter.h"
#include "EngineInterface/GenomeDescriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class DetonatorTests : public IntegrationTestFramework
{
public:
    static SimulationParameters getParameters()
    {
        SimulationParameters result;
        result.innerFriction = 0;
        result.baseValues.friction = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.baseValues.radiationCellAgeStrength[i] = 0;
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
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
            .setCellFunction(DetonatorDescription().setCountDown(14)),
    });

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualDetonatorCell = getCell(actualData, 1);

    EXPECT_EQ(1, actualData.cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(0.0f, actualDetonatorCell.activity.channels[0]));
    EXPECT_EQ(14, std::get<DetonatorDescription>(*actualDetonatorCell.cellFunction).countdown);
    EXPECT_EQ(DetonatorState_Ready, std::get<DetonatorDescription>(*actualDetonatorCell.cellFunction).state);
}

TEST_F(DetonatorTests, activateDetonator)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(DetonatorDescription().setCountDown(10)),
         CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualDetonatorCell = getCell(actualData, 1);

    EXPECT_EQ(2, actualData.cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(1.0f, actualDetonatorCell.activity.channels[0]));
    EXPECT_EQ(9, std::get<DetonatorDescription>(*actualDetonatorCell.cellFunction).countdown);
    EXPECT_EQ(DetonatorState_Activated, std::get<DetonatorDescription>(*actualDetonatorCell.cellFunction).state);
}

