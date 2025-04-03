#include <cmath>
#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class MuscleTests : public IntegrationTestFramework
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

    MuscleTests()
        : IntegrationTestFramework(getParameters())
    {
    }

    ~MuscleTests() = default;
};

//TEST_F(MuscleTests, doNothing)
//{
//    DataDescription data;
//    data.addCells(
//        {CellDescription()
//             .id(1)
//             .pos({10.0f, 10.0f})
//             .cellType(MuscleDescription().mode(MuscleMode_Movement)),
//         CellDescription()
//             .id(2)
//             .pos({11.0f, 10.0f})
//             .cellType(OscillatorDescription())
//             .signal({0, 0, 0, 0, 0, 0, 0, 0})});
//    data.addConnection(1, 2);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//    auto actualMuscleCell = getCell(actualData, 1);
//    auto actualOscillatorCell = getCell(actualData, 2);
//
//    EXPECT_EQ(2, actualData._cells.size());
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//    EXPECT_TRUE(approxCompare(0, actualMuscleCell._signal->_channels[0]));
//    EXPECT_TRUE(approxCompare(actualOscillatorCell._connections.at(0)._distance, actualMuscleCell._connections.at(0)._distance));
//    EXPECT_TRUE(approxCompare(1.0f , actualMuscleCell._connections.at(0)._distance));
//    EXPECT_TRUE(approxCompare(0, actualMuscleCell._vel.x));
//    EXPECT_TRUE(approxCompare(0, actualMuscleCell._vel.y));
//}
//
//TEST_F(MuscleTests, moveForward)
//{
//    DataDescription data;
//    data.addCells(
//        {CellDescription()
//             .id(1)
//             .pos({10.0f, 10.0f})
//             .cellType(MuscleDescription().mode(MuscleMode_Movement)),
//         CellDescription()
//             .id(2)
//             .pos({11.0f, 10.0f})
//             .cellType(OscillatorDescription())
//             .signal({1, 0, 0, 0, 0, 0, 0, 0})});
//    data.addConnection(1, 2);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//    auto actualMuscleCell = getCell(actualData, 1);
//    auto actualOscillatorCell = getCell(actualData, 2);
//
//    EXPECT_EQ(2, actualData._cells.size());
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._signal->_channels[0]));
//    EXPECT_TRUE(approxCompare(actualOscillatorCell._connections.at(0)._distance, actualMuscleCell._connections.at(0)._distance));
//    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));
//    EXPECT_TRUE(approxCompare(-_parameters.cellTypeMuscleMovementAcceleration[0], actualMuscleCell._vel.x));
//    EXPECT_TRUE(approxCompare(0, actualMuscleCell._vel.y));
//}
//
//TEST_F(MuscleTests, moveBackward)
//{
//    DataDescription data;
//    data.addCells(
//        {CellDescription()
//             .id(1)
//             .pos({10.0f, 10.0f})
//             .cellType(MuscleDescription().mode(MuscleMode_Movement)),
//         CellDescription()
//             .id(2)
//             .pos({11.0f, 10.0f})
//             .cellType(OscillatorDescription())
//             .signal({-1, 0, 0, 0, 0, 0, 0, 0})});
//    data.addConnection(1, 2);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//    auto actualMuscleCell = getCell(actualData, 1);
//    auto actualOscillatorCell = getCell(actualData, 2);
//
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//    EXPECT_TRUE(approxCompare(-1.0f, actualMuscleCell._signal->_channels[0]));
//    EXPECT_TRUE(approxCompare(actualOscillatorCell._connections.at(0)._distance, actualMuscleCell._connections.at(0)._distance));
//    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));
//    EXPECT_TRUE(approxCompare(_parameters.cellTypeMuscleMovementAcceleration[0], actualMuscleCell._vel.x));
//    EXPECT_TRUE(approxCompare(0, actualMuscleCell._vel.y));
//}
//
//TEST_F(MuscleTests, multipleMovementDirections)
//{
//    DataDescription data;
//    data.addCells({
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .cellType(MuscleDescription().mode(MuscleMode_Movement)),
//        CellDescription()
//            .id(2)
//            .pos({11.0f, 10.0f})
//            .cellType(OscillatorDescription())
//            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
//        CellDescription()
//            .id(3)
//            .pos({10.0f, 11.0f})
//            .cellType(OscillatorDescription())
//            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(1, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//    auto actualMuscleCell = getCell(actualData, 1);
//
//    EXPECT_TRUE(approxCompare(2.0f, actualMuscleCell._signal->_channels[0]));
//    EXPECT_TRUE(actualMuscleCell._vel.x < -NEAR_ZERO);
//    EXPECT_TRUE(actualMuscleCell._vel.y < -NEAR_ZERO);
//}
//
//TEST_F(MuscleTests, expansion)
//{
//    auto const smallDistance = _parameters.cellMinDistance * 1.1f;
//
//    DataDescription data;
//    data.addCells(
//        {CellDescription()
//             .id(1)
//             .pos({10.0f, 10.0f})
//             .cellType(MuscleDescription().mode(MuscleMode_ContractionExpansion)),
//         CellDescription()
//             .id(2)
//             .pos({10.0f + smallDistance, 10.0f})
//             .cellType(OscillatorDescription())
//             .signal({1, 0, 0, 0, 0, 0, 0, 0})});
//    data.addConnection(1, 2);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//    auto actualMuscleCell = getCell(actualData, 1);
//    auto actualOscillatorCell = getCell(actualData, 2);
//
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._signal->_channels[0]));
//    EXPECT_TRUE(approxCompare(actualOscillatorCell._connections.at(0)._distance, actualMuscleCell._connections.at(0)._distance));
//    EXPECT_TRUE(approxCompare(smallDistance + _parameters.cellTypeMuscleContractionExpansionDelta[0], actualMuscleCell._connections.at(0)._distance));
//    EXPECT_TRUE(approxCompare(0, actualMuscleCell._vel.x));
//    EXPECT_TRUE(approxCompare(0, actualMuscleCell._vel.y));
//}
//
//TEST_F(MuscleTests, expansionNotPossible)
//{
//    auto const largeDistance = _parameters.cellMaxBindingDistance[0] * 0.9f;
//
//    DataDescription data;
//    data.addCells(
//        {CellDescription()
//             .id(1)
//             .pos({10.0f, 10.0f})
//             .cellType(MuscleDescription().mode(MuscleMode_ContractionExpansion)),
//         CellDescription()
//             .id(2)
//             .pos({10.0f + largeDistance, 10.0f})
//             .cellType(OscillatorDescription())
//             .signal({1, 0, 0, 0, 0, 0, 0, 0})});
//    data.addConnection(1, 2);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//    auto actualMuscleCell = getCell(actualData, 1);
//    auto actualOscillatorCell = getCell(actualData, 2);
//
//    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._signal->_channels[0]));
//    EXPECT_TRUE(approxCompare(largeDistance, actualMuscleCell._connections.at(0)._distance));
//}
//
//TEST_F(MuscleTests, contraction)
//{
//    auto const largeDistance = _parameters.cellMaxBindingDistance[0] * 0.9f;
//
//    DataDescription data;
//    data.addCells(
//        {CellDescription()
//             .id(1)
//             .pos({10.0f, 10.0f})
//             .cellType(MuscleDescription().mode(MuscleMode_ContractionExpansion)),
//         CellDescription()
//             .id(2)
//             .pos({10.0f + largeDistance, 10.0f})
//             .cellType(OscillatorDescription())
//             .signal({-1, 0, 0, 0, 0, 0, 0, 0})});
//    data.addConnection(1, 2);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//    auto actualMuscleCell = getCell(actualData, 1);
//    auto actualOscillatorCell = getCell(actualData, 2);
//
//    EXPECT_TRUE(approxCompare(-1.0f, actualMuscleCell._signal->_channels[0]));
//    EXPECT_TRUE(approxCompare(actualOscillatorCell._connections.at(0)._distance, actualMuscleCell._connections.at(0)._distance));
//    EXPECT_TRUE(approxCompare(largeDistance - _parameters.cellTypeMuscleContractionExpansionDelta[0], actualMuscleCell._connections.at(0)._distance));
//}
//
//TEST_F(MuscleTests, multipleContraction)
//{
//    auto const largeDistance = _parameters.cellMaxBindingDistance[0] * 0.9f;
//
//    DataDescription data;
//    data.addCells({
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .cellType(MuscleDescription().mode(MuscleMode_ContractionExpansion)),
//        CellDescription()
//            .id(2)
//            .pos({10.0f + largeDistance, 10.0f})
//            .cellType(OscillatorDescription())
//            .signal({-1, 0, 0, 0, 0, 0, 0, 0}),
//        CellDescription()
//            .id(3)
//            .pos({10.0f, 10.0f + largeDistance})
//            .cellType(OscillatorDescription())
//            .signal({-1, 0, 0, 0, 0, 0, 0, 0}),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(1, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//    auto actualMuscleCell = getCell(actualData, 1);
//    auto muscleToOscillatorConnection1 = getConnection(actualData, 1, 2);
//    auto muscleToOscillatorConnection2 = getConnection(actualData, 1, 3);
//
//    EXPECT_TRUE(approxCompare(-2.0f, actualMuscleCell._signal->_channels[0]));
//    EXPECT_TRUE(approxCompare(largeDistance - _parameters.cellTypeMuscleContractionExpansionDelta[0], muscleToOscillatorConnection1._distance));
//    EXPECT_TRUE(approxCompare(largeDistance - _parameters.cellTypeMuscleContractionExpansionDelta[0], muscleToOscillatorConnection2._distance));
//}
//
//TEST_F(MuscleTests, contractionNotPossible)
//{
//    auto const smallDistance = _parameters.cellMinDistance * 1.1f;
//
//    DataDescription data;
//    data.addCells(
//        {CellDescription()
//             .id(1)
//             .pos({10.0f, 10.0f})
//             .cellType(MuscleDescription().mode(MuscleMode_ContractionExpansion)),
//         CellDescription()
//             .id(2)
//             .pos({10.0f + smallDistance, 10.0f})
//             .cellType(OscillatorDescription())
//             .signal({-1, 0, 0, 0, 0, 0, 0, 0})});
//    data.addConnection(1, 2);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//    auto actualMuscleCell = getCell(actualData, 1);
//    auto actualOscillatorCell = getCell(actualData, 2);
//
//    EXPECT_TRUE(approxCompare(-1.0f, actualMuscleCell._signal->_channels[0]));
//    EXPECT_TRUE(approxCompare(smallDistance, actualMuscleCell._connections.at(0)._distance));
//}
//
//TEST_F(MuscleTests, bendClockwise)
//{
//    DataDescription data;
//    data.addCells({
//        CellDescription().id(1).pos({9.0f, 10.0f}),
//        CellDescription()
//            .id(2)
//            .pos({10.0f, 10.0f})
//            .cellType(MuscleDescription().mode(MuscleMode_Bending)),
//        CellDescription()
//            .id(3)
//            .pos({11.0f, 10.0f})
//            .cellType(OscillatorDescription())
//            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//    auto actualMuscleCell = getCell(actualData, 2);
//    auto actualOscillatorCell = getCell(actualData, 3);
//    auto connection1 = getConnection(actualData, 2, 3);
//    auto connection2 = getConnection(actualData, 3, 2);
//
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._signal->_channels[0]));
//    EXPECT_TRUE(approxCompare(1.0f, connection1._distance));
//    EXPECT_TRUE(approxCompare(1.0f, connection2._distance));
//    EXPECT_TRUE(approxCompare(180.0f + _parameters.cellTypeMuscleBendingAngle[0], connection1._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(0, actualMuscleCell._vel.x));
//    EXPECT_TRUE(approxCompare(0, actualMuscleCell._vel.y));
//}
//
//TEST_F(MuscleTests, bendCounterClockwise)
//{
//    DataDescription data;
//    data.addCells({
//        CellDescription().id(1).pos({9.0f, 10.0f}),
//        CellDescription()
//            .id(2)
//            .pos({10.0f, 10.0f})
//            .cellType(MuscleDescription().mode(MuscleMode_Bending)),
//        CellDescription()
//            .id(3)
//            .pos({11.0f, 10.0f})
//            .cellType(OscillatorDescription())
//            .signal({-1, 0, 0, 0, 0, 0, 0, 0}),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//    auto actualMuscleCell = getCell(actualData, 2);
//    auto actualOscillatorCell = getCell(actualData, 3);
//    auto connection1 = getConnection(actualData, 2, 3);
//    auto connection2 = getConnection(actualData, 3, 2);
//
//    EXPECT_TRUE(approxCompare(-1.0f, actualMuscleCell._signal->_channels[0]));
//    EXPECT_TRUE(approxCompare(1.0f, connection1._distance));
//    EXPECT_TRUE(approxCompare(1.0f, connection2._distance));
//    EXPECT_TRUE(approxCompare(180.0f - _parameters.cellTypeMuscleBendingAngle[0], connection1._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(0, actualMuscleCell._vel.x));
//    EXPECT_TRUE(approxCompare(0, actualMuscleCell._vel.y));
//}
