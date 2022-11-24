#include <gtest/gtest.h>

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class AttackerTests : public IntegrationTestFramework
{
public:
    AttackerTests()
        : IntegrationTestFramework()
    {}

    ~AttackerTests() = default;
};

TEST_F(AttackerTests, nothingFound)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setCellFunction(AttackerDescription()),
         CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();

    auto actualData = _simController->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_EQ(2, actualData.cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell.activity.channels[0]));
}

TEST_F(AttackerTests, success)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({10.0f, 10.0f}).setMaxConnections(2).setExecutionOrderNumber(0).setCellFunction(AttackerDescription()),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setActivity({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription()
            .setId(3)
            .setPos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();

    auto actualData = _simController->getSimulationData();
    auto origAttackCell = getCell(data, 1);
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualAttackCell.activity.channels[0]));
    EXPECT_TRUE(actualAttackCell.energy > origAttackCell.energy + FLOATINGPOINT_MEDIUM_PRECISION);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}
