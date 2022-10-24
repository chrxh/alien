#include <gtest/gtest.h>

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class NerveTests : public IntegrationTestFramework
{
public:
    NerveTests()
        : IntegrationTestFramework({1000, 1000})
    {}

    ~NerveTests() = default;
};

TEST_F(NerveTests, noInput_execution)
{
    ActivityDescription activity;
    activity.channels = {1, 0, -1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputBlocked(true)
            .setActivity(activity),
    });

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    EXPECT_EQ(ActivityDescription(), actualCellById.at(1).activity);
    EXPECT_TRUE(actualCellById.at(1).activityChanged);
}

TEST_F(NerveTests, noInput_noExecution)
{
    ActivityDescription activity;
    activity.channels = {1, 0, -1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(1)
            .setInputBlocked(true)
            .setActivity(activity),
    });

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    EXPECT_EQ(activity, actualCellById.at(1).activity);
    EXPECT_FALSE(actualCellById.at(1).activityChanged);
}

TEST_F(NerveTests, activityTransfer)
{
    ActivityDescription activity;
    activity.channels = {1, 0, 1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setPos({1.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setInputBlocked(true)
            .setActivity(activity),
        CellDescription()
            .setId(2)
            .setPos({2.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(0),
        CellDescription()
            .setId(3)
            .setPos({3.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(1)
            .setOutputBlocked(true),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();

    {
        auto actualData = _simController->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(activity, actualCellById.at(1).activity);
        EXPECT_FALSE(actualCellById.at(1).activityChanged);

        EXPECT_EQ(activity, actualCellById.at(2).activity);
        EXPECT_TRUE(actualCellById.at(2).activityChanged);

        EXPECT_EQ(ActivityDescription(), actualCellById.at(3).activity);
        EXPECT_FALSE(actualCellById.at(3).activityChanged);
    }

    for (int i = 0; i < 5; ++i) {
        _simController->calcSingleTimestep();
    }
    {
        auto actualData = _simController->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(ActivityDescription(), actualCellById.at(1).activity);
        EXPECT_TRUE(actualCellById.at(1).activityChanged);

        EXPECT_EQ(activity, actualCellById.at(2).activity);
        EXPECT_TRUE(actualCellById.at(2).activityChanged);

        EXPECT_EQ(activity, actualCellById.at(3).activity);
        EXPECT_TRUE(actualCellById.at(3).activityChanged);
    }

    for (int i = 0; i < 6; ++i) {
        _simController->calcSingleTimestep();
    }
    {
        auto actualData = _simController->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(ActivityDescription(), actualCellById.at(1).activity);
        EXPECT_FALSE(actualCellById.at(1).activityChanged);

        EXPECT_EQ(ActivityDescription(), actualCellById.at(2).activity);
        EXPECT_TRUE(actualCellById.at(2).activityChanged);

        EXPECT_EQ(ActivityDescription(), actualCellById.at(3).activity);
        EXPECT_TRUE(actualCellById.at(3).activityChanged);
    }

    for (int i = 0; i < 6; ++i) {
        _simController->calcSingleTimestep();
    }
    {
        auto actualData = _simController->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(ActivityDescription(), actualCellById.at(1).activity);
        EXPECT_FALSE(actualCellById.at(1).activityChanged);

        EXPECT_EQ(ActivityDescription(), actualCellById.at(2).activity);
        EXPECT_FALSE(actualCellById.at(2).activityChanged);

        EXPECT_EQ(ActivityDescription(), actualCellById.at(3).activity);
        EXPECT_FALSE(actualCellById.at(3).activityChanged);
    }
}
