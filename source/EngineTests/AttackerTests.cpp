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

TEST_F(AttackerTests, successNoTransmitter)
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

TEST_F(AttackerTests, successOneTransmitter)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({10.0f, 10.0f}).setMaxConnections(1).setExecutionOrderNumber(0).setCellFunction(AttackerDescription()),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setActivity({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({12.0f, 10.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()),
        CellDescription().setId(4).setPos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();

    auto actualData = _simController->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);
    auto origTransmitterCell = getCell(data, 3);
    auto actualTransmitterCell = getCell(actualData, 3);

    EXPECT_TRUE(approxCompare(1.0f, actualAttackCell.activity.channels[0]));
    EXPECT_TRUE(actualTransmitterCell.energy > origTransmitterCell.energy + FLOATINGPOINT_MEDIUM_PRECISION);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(AttackerTests, successTwoTransmitters)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({10.0f, 10.0f}).setMaxConnections(2).setExecutionOrderNumber(0).setCellFunction(AttackerDescription()),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setActivity({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({12.0f, 10.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()),
        CellDescription().setId(4).setPos({11.0f, 9.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()),
        CellDescription().setId(5).setPos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 4);
    data.addConnection(2, 3);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();

    auto actualData = _simController->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);
    auto origTransmitterCell1 = getCell(data, 3);
    auto actualTransmitterCell1 = getCell(actualData, 3);
    auto origTransmitterCell2 = getCell(data, 4);
    auto actualTransmitterCell2 = getCell(actualData, 4);

    EXPECT_TRUE(approxCompare(1.0f, actualAttackCell.activity.channels[0]));
    EXPECT_TRUE(actualTransmitterCell1.energy > origTransmitterCell1.energy + FLOATINGPOINT_MEDIUM_PRECISION);
    EXPECT_TRUE(actualTransmitterCell2.energy > origTransmitterCell2.energy + FLOATINGPOINT_MEDIUM_PRECISION);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}
