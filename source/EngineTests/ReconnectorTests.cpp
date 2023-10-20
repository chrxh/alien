#include <gtest/gtest.h>

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionConverter.h"
#include "EngineInterface/GenomeDescriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class ReconnectorTests : public IntegrationTestFramework
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
    ReconnectorTests()
        : IntegrationTestFramework(getParameters())
    {}

    ~ReconnectorTests() = default;
};

TEST_F(ReconnectorTests, nothingFound)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(ReconnectorDescription()),
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
    auto actualReconnectorCell = getCell(actualData, 1);

    EXPECT_EQ(2, actualData.cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(0.0f, actualReconnectorCell.activity.channels[0]));
}

TEST_F(ReconnectorTests, success)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(ReconnectorDescription()),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setActivity({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);

    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualReconnectorCell.activity.channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell.connections.size());
    EXPECT_EQ(1, actualTargetCell.connections.size());
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}
