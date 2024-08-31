#include <cmath>
#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class MuscleTests : public IntegrationTestFramework
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

    MuscleTests()
        : IntegrationTestFramework(getParameters())
    {
    }

    ~MuscleTests() = default;
};

TEST_F(MuscleTests, doNothing)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(MuscleDescription().setMode(MuscleMode_Movement)),
         CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({0, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualMuscleCell = getCell(actualData, 1);
    auto actualNerveCell = getCell(actualData, 2);

    EXPECT_EQ(2, actualData.cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(0, actualMuscleCell.activity.channels[0]));
    EXPECT_TRUE(approxCompare(actualNerveCell.connections.at(0).distance, actualMuscleCell.connections.at(0).distance));
    EXPECT_TRUE(approxCompare(1.0f , actualMuscleCell.connections.at(0).distance));
    EXPECT_TRUE(approxCompare(0, actualMuscleCell.vel.x));
    EXPECT_TRUE(approxCompare(0, actualMuscleCell.vel.y));
}

TEST_F(MuscleTests, moveForward)
{
    _parameters.features.legacyModes = true;
    _parameters.legacyCellFunctionMuscleMovementModeActivated = true;
    _simController->setSimulationParameters(_parameters);

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(MuscleDescription().setMode(MuscleMode_Movement)),
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
    auto actualMuscleCell = getCell(actualData, 1);
    auto actualNerveCell = getCell(actualData, 2);

    EXPECT_EQ(2, actualData.cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell.activity.channels[0]));
    EXPECT_TRUE(approxCompare(actualNerveCell.connections.at(0).distance, actualMuscleCell.connections.at(0).distance));
    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell.connections.at(0).distance));
    EXPECT_TRUE(approxCompare(-_parameters.cellFunctionMuscleMovementAcceleration[0], actualMuscleCell.vel.x));
    EXPECT_TRUE(approxCompare(0, actualMuscleCell.vel.y));
}

TEST_F(MuscleTests, moveBackward)
{
    _parameters.features.legacyModes = true;
    _parameters.legacyCellFunctionMuscleMovementModeActivated = true;
    _simController->setSimulationParameters(_parameters);

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(MuscleDescription().setMode(MuscleMode_Movement)),
         CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({-1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualMuscleCell = getCell(actualData, 1);
    auto actualNerveCell = getCell(actualData, 2);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(-1.0f, actualMuscleCell.activity.channels[0]));
    EXPECT_TRUE(approxCompare(actualNerveCell.connections.at(0).distance, actualMuscleCell.connections.at(0).distance));
    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell.connections.at(0).distance));
    EXPECT_TRUE(approxCompare(_parameters.cellFunctionMuscleMovementAcceleration[0], actualMuscleCell.vel.x));
    EXPECT_TRUE(approxCompare(0, actualMuscleCell.vel.y));
}

TEST_F(MuscleTests, multipleMovementDirections)
{
    _parameters.features.legacyModes = true;
    _parameters.legacyCellFunctionMuscleMovementModeActivated = true;
    _simController->setSimulationParameters(_parameters);

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(MuscleDescription().setMode(MuscleMode_Movement)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setActivity({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription()
            .setId(3)
            .setPos({10.0f, 11.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setActivity({1, 0, 0, 0, 0, 0, 0, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualMuscleCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(2.0f, actualMuscleCell.activity.channels[0]));
    EXPECT_TRUE(actualMuscleCell.vel.x < -NEAR_ZERO);
    EXPECT_TRUE(actualMuscleCell.vel.y < -NEAR_ZERO);
}

TEST_F(MuscleTests, expansion)
{
    auto const smallDistance = _parameters.cellMinDistance * 1.1f;

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(MuscleDescription().setMode(MuscleMode_ContractionExpansion)),
         CellDescription()
             .setId(2)
             .setPos({10.0f + smallDistance, 10.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualMuscleCell = getCell(actualData, 1);
    auto actualNerveCell = getCell(actualData, 2);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell.activity.channels[0]));
    EXPECT_TRUE(approxCompare(actualNerveCell.connections.at(0).distance, actualMuscleCell.connections.at(0).distance));
    EXPECT_TRUE(approxCompare(smallDistance + _parameters.cellFunctionMuscleContractionExpansionDelta[0], actualMuscleCell.connections.at(0).distance));
    EXPECT_TRUE(approxCompare(0, actualMuscleCell.vel.x));
    EXPECT_TRUE(approxCompare(0, actualMuscleCell.vel.y));
}

TEST_F(MuscleTests, expansionNotPossible)
{
    auto const largeDistance = _parameters.cellMaxBindingDistance[0] * 0.9f;

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(MuscleDescription().setMode(MuscleMode_ContractionExpansion)),
         CellDescription()
             .setId(2)
             .setPos({10.0f + largeDistance, 10.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualMuscleCell = getCell(actualData, 1);
    auto actualNerveCell = getCell(actualData, 2);

    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell.activity.channels[0]));
    EXPECT_TRUE(approxCompare(largeDistance, actualMuscleCell.connections.at(0).distance));
}

TEST_F(MuscleTests, contraction)
{
    auto const largeDistance = _parameters.cellMaxBindingDistance[0] * 0.9f;

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(MuscleDescription().setMode(MuscleMode_ContractionExpansion)),
         CellDescription()
             .setId(2)
             .setPos({10.0f + largeDistance, 10.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({-1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualMuscleCell = getCell(actualData, 1);
    auto actualNerveCell = getCell(actualData, 2);

    EXPECT_TRUE(approxCompare(-1.0f, actualMuscleCell.activity.channels[0]));
    EXPECT_TRUE(approxCompare(actualNerveCell.connections.at(0).distance, actualMuscleCell.connections.at(0).distance));
    EXPECT_TRUE(approxCompare(largeDistance - _parameters.cellFunctionMuscleContractionExpansionDelta[0], actualMuscleCell.connections.at(0).distance));
}

TEST_F(MuscleTests, multipleContraction)
{
    auto const largeDistance = _parameters.cellMaxBindingDistance[0] * 0.9f;

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(MuscleDescription().setMode(MuscleMode_ContractionExpansion)),
        CellDescription()
            .setId(2)
            .setPos({10.0f + largeDistance, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setActivity({-1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription()
            .setId(3)
            .setPos({10.0f, 10.0f + largeDistance})
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setActivity({-1, 0, 0, 0, 0, 0, 0, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualMuscleCell = getCell(actualData, 1);
    auto muscleToNerveConnection1 = getConnection(actualData, 1, 2);
    auto muscleToNerveConnection2 = getConnection(actualData, 1, 3);

    EXPECT_TRUE(approxCompare(-2.0f, actualMuscleCell.activity.channels[0]));
    EXPECT_TRUE(approxCompare(largeDistance - _parameters.cellFunctionMuscleContractionExpansionDelta[0], muscleToNerveConnection1.distance));
    EXPECT_TRUE(approxCompare(largeDistance - _parameters.cellFunctionMuscleContractionExpansionDelta[0], muscleToNerveConnection2.distance));
}

TEST_F(MuscleTests, contractionNotPossible)
{
    auto const smallDistance = _parameters.cellMinDistance * 1.1f;

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(MuscleDescription().setMode(MuscleMode_ContractionExpansion)),
         CellDescription()
             .setId(2)
             .setPos({10.0f + smallDistance, 10.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({-1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualMuscleCell = getCell(actualData, 1);
    auto actualNerveCell = getCell(actualData, 2);

    EXPECT_TRUE(approxCompare(-1.0f, actualMuscleCell.activity.channels[0]));
    EXPECT_TRUE(approxCompare(smallDistance, actualMuscleCell.connections.at(0).distance));
}

TEST_F(MuscleTests, bendClockwise)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({9.0f, 10.0f}).setMaxConnections(1),
        CellDescription()
            .setId(2)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(MuscleDescription().setMode(MuscleMode_Bending)),
        CellDescription()
            .setId(3)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setActivity({1, 0, 0, 0, 0, 0, 0, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualMuscleCell = getCell(actualData, 2);
    auto actualNerveCell = getCell(actualData, 3);
    auto connection1 = getConnection(actualData, 2, 3);
    auto connection2 = getConnection(actualData, 3, 2);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell.activity.channels[0]));
    EXPECT_TRUE(approxCompare(1.0f, connection1.distance));
    EXPECT_TRUE(approxCompare(1.0f, connection2.distance));
    EXPECT_TRUE(approxCompare(180.0f + _parameters.cellFunctionMuscleBendingAngle[0], connection1.angleFromPrevious));
    EXPECT_TRUE(approxCompare(0, actualMuscleCell.vel.x));
    EXPECT_TRUE(approxCompare(0, actualMuscleCell.vel.y));
}

TEST_F(MuscleTests, bendCounterClockwise)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({9.0f, 10.0f}).setMaxConnections(1),
        CellDescription()
            .setId(2)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(MuscleDescription().setMode(MuscleMode_Bending)),
        CellDescription()
            .setId(3)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setActivity({-1, 0, 0, 0, 0, 0, 0, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualMuscleCell = getCell(actualData, 2);
    auto actualNerveCell = getCell(actualData, 3);
    auto connection1 = getConnection(actualData, 2, 3);
    auto connection2 = getConnection(actualData, 3, 2);

    EXPECT_TRUE(approxCompare(-1.0f, actualMuscleCell.activity.channels[0]));
    EXPECT_TRUE(approxCompare(1.0f, connection1.distance));
    EXPECT_TRUE(approxCompare(1.0f, connection2.distance));
    EXPECT_TRUE(approxCompare(180.0f - _parameters.cellFunctionMuscleBendingAngle[0], connection1.angleFromPrevious));
    EXPECT_TRUE(approxCompare(0, actualMuscleCell.vel.x));
    EXPECT_TRUE(approxCompare(0, actualMuscleCell.vel.y));
}
