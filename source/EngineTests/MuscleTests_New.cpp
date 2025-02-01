#include <cmath>

#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class MuscleTests_New : public IntegrationTestFramework
{
public:
    MuscleTests_New()
        : IntegrationTestFramework()
    {}

    ~MuscleTests_New() = default;
};

TEST_F(MuscleTests_New, autobending_backAndForth_rightSide)
{
    auto constexpr MaxAngleDeviation = 30.0f;

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellType(OscillatorDescription().autoTriggerInterval(20)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .absAngleToConnection0(90.0f)
            .cellType(MuscleDescription().mode(AutoBendingDescription().maxAngleDeviation(MaxAngleDeviation))),
        CellDescription().id(3).pos({12.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);

    auto minAngle = std::numeric_limits<float>::max();
    auto maxAngle = std::numeric_limits<float>::min();
    for (int i = 0; i < 200; ++i) {
        _simulationFacade->calcTimesteps(10);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = getCell(actualData, 2);
        auto actualCell1 = getCell(actualData, 1);
        auto actualCell3 = getCell(actualData, 3);

        EXPECT_EQ(3, actualData._cells.size());

        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(1)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));

        minAngle = std::min(minAngle, actualMuscleCell._connections.at(0)._angleFromPrevious);
        maxAngle = std::max(maxAngle, actualMuscleCell._connections.at(0)._angleFromPrevious);
        if (i == 0) {
            EXPECT_TRUE(actualMuscleCell._connections.at(0)._angleFromPrevious < 180.0f - NEAR_ZERO);
        }
    }
    EXPECT_TRUE(minAngle < 180.0f - MaxAngleDeviation + NEAR_ZERO);
    EXPECT_TRUE(minAngle > 180.0f - MaxAngleDeviation - NEAR_ZERO);
    EXPECT_TRUE(maxAngle > 180.0f + MaxAngleDeviation - NEAR_ZERO);
    EXPECT_TRUE(maxAngle < 180.0f + MaxAngleDeviation + NEAR_ZERO);
}

TEST_F(MuscleTests_New, autobending_backAndForth_leftSide)
{
    auto constexpr MaxAngleDeviation = 30.0f;

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellType(OscillatorDescription().autoTriggerInterval(20)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .absAngleToConnection0(-90.0f)
            .cellType(MuscleDescription().mode(AutoBendingDescription().maxAngleDeviation(MaxAngleDeviation))),
        CellDescription().id(3).pos({12.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);

    auto minAngle = std::numeric_limits<float>::max();
    auto maxAngle = std::numeric_limits<float>::min();
    for (int i = 0; i < 200; ++i) {
        _simulationFacade->calcTimesteps(10);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = getCell(actualData, 2);
        auto actualCell1 = getCell(actualData, 1);
        auto actualCell3 = getCell(actualData, 3);

        EXPECT_EQ(3, actualData._cells.size());

        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(1)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));

        minAngle = std::min(minAngle, actualMuscleCell._connections.at(0)._angleFromPrevious);
        maxAngle = std::max(maxAngle, actualMuscleCell._connections.at(0)._angleFromPrevious);
        if (i == 0) {
            EXPECT_TRUE(actualMuscleCell._connections.at(0)._angleFromPrevious > 180.0f + NEAR_ZERO);
        }
    }
    EXPECT_TRUE(minAngle < 180.0f - MaxAngleDeviation + NEAR_ZERO);
    EXPECT_TRUE(minAngle > 180.0f - MaxAngleDeviation - NEAR_ZERO);
    EXPECT_TRUE(maxAngle > 180.0f + MaxAngleDeviation - NEAR_ZERO);
    EXPECT_TRUE(maxAngle < 180.0f + MaxAngleDeviation + NEAR_ZERO);
}
