#include <cmath>

#include <gtest/gtest.h>

#include "Base/Math.h"
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

protected:
    template <typename Channel>
    float getValue(Channel channel)
    {
        if (channel == Channel::Positive) {
            return 1.0f;
        } else if (channel == Channel::Negative) {
            return -1.0f;
        } else {
            return 0.0f;
        }
    }
};

enum class Side
{
    Left,
    Right
};

enum class Channel0
{
    Zero,
    Positive,
    Negative
};

enum class Channel1
{
    Zero,
    Positive,
    Negative
};

class MuscleTests_AutoBending_New
    : public MuscleTests_New
    , public testing::WithParamInterface<std::tuple<Side, Channel0, Channel1>>
{
};

INSTANTIATE_TEST_SUITE_P(
    MuscleTests_AutoBending_New,
    MuscleTests_AutoBending_New,
    ::testing::Values(
        std::make_tuple(Side::Left, Channel0::Zero, Channel1::Zero),
        std::make_tuple(Side::Right, Channel0::Zero, Channel1::Zero),
        std::make_tuple(Side::Left, Channel0::Positive, Channel1::Zero),
        std::make_tuple(Side::Right, Channel0::Positive, Channel1::Zero),
        std::make_tuple(Side::Left, Channel0::Negative, Channel1::Zero),
        std::make_tuple(Side::Right, Channel0::Negative, Channel1::Zero),
        std::make_tuple(Side::Left, Channel0::Zero, Channel1::Positive),
        std::make_tuple(Side::Right, Channel0::Zero, Channel1::Positive),
        std::make_tuple(Side::Left, Channel0::Positive, Channel1::Positive),
        std::make_tuple(Side::Right, Channel0::Positive, Channel1::Positive),
        std::make_tuple(Side::Left, Channel0::Negative, Channel1::Positive),
        std::make_tuple(Side::Right, Channel0::Negative, Channel1::Positive),
        std::make_tuple(Side::Left, Channel0::Zero, Channel1::Negative),
        std::make_tuple(Side::Right, Channel0::Zero, Channel1::Negative),
        std::make_tuple(Side::Left, Channel0::Positive, Channel1::Negative),
        std::make_tuple(Side::Right, Channel0::Positive, Channel1::Negative),
        std::make_tuple(Side::Left, Channel0::Negative, Channel1::Negative),
        std::make_tuple(Side::Right, Channel0::Negative, Channel1::Negative)));

TEST_P(MuscleTests_AutoBending_New, muscleWithTwoConnections)
{
    auto constexpr MaxAngleDeviation = 30.0f;
    auto constexpr AnglePrecision = NEAR_ZERO;

    auto [side, channel0, channel1] = GetParam();

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({side == Side::Left ? 10.0f : 12.0f, 10.0f}).cellType(OscillatorDescription().autoTriggerInterval(20)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .angleToFront(side == Side::Left ? 90.0f : -90.0f)
            .cellType(MuscleDescription().mode(AutoBendingDescription().maxAngleDeviation(MaxAngleDeviation * 2 / 180.0f)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, getValue(channel0)).weight(1, 0, getValue(channel1) / 4)),
        CellDescription().id(3).pos({side == Side::Left ? 12.0f : 10.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);

    auto minAngle = 180.0f;
    auto maxAngle = 180.0f;
    for (int i = 0; i < 200; ++i) {
        _simulationFacade->calcTimesteps(10);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = getCell(actualData, 2);
        auto actualCell1 = getCell(actualData, 1);
        auto actualCell3 = getCell(actualData, 3);

        ASSERT_EQ(3, actualData._cells.size());

        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(1)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));

        auto angle = actualMuscleCell._connections.at(0)._angleFromPrevious;
        minAngle = std::min(minAngle, angle);
        maxAngle = std::max(maxAngle, angle);
        if (i == 0) {
            if (channel0 == Channel0::Zero) {
                EXPECT_TRUE(angle < 180.0f + NEAR_ZERO);
                EXPECT_TRUE(angle > 180.0f - NEAR_ZERO);
            } else if ((side == Side::Left && channel0 == Channel0::Positive) || (side == Side::Right && channel0 == Channel0::Negative)) {
                EXPECT_TRUE(angle > 180.0f + NEAR_ZERO);
            } else {
                EXPECT_TRUE(angle < 180.0f - NEAR_ZERO);
            }
        }
    }

    if (channel0 == Channel0::Zero) {
        EXPECT_TRUE(minAngle < 180.0f + AnglePrecision);
        EXPECT_TRUE(minAngle > 180.0f - AnglePrecision);
        EXPECT_TRUE(maxAngle < 180.0f + AnglePrecision);
        EXPECT_TRUE(maxAngle > 180.0f - AnglePrecision);
    } else if (channel1 == Channel1::Zero) {
        EXPECT_TRUE(minAngle < 180.0f - MaxAngleDeviation + AnglePrecision);
        EXPECT_TRUE(minAngle > 180.0f - MaxAngleDeviation - AnglePrecision);
        EXPECT_TRUE(maxAngle > 180.0f + MaxAngleDeviation - AnglePrecision);
        EXPECT_TRUE(maxAngle < 180.0f + MaxAngleDeviation + AnglePrecision);
    }
}

TEST_P(MuscleTests_AutoBending_New, muscleWithOneConnection)
{
    auto constexpr MaxAngleDeviation = 30.0f;
    auto constexpr AnglePrecision = NEAR_ZERO;

    auto [side, channel0, channel1] = GetParam();

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}),
        CellDescription().id(2).pos({10.0f, 11.0f}).cellType(OscillatorDescription().autoTriggerInterval(20)),
        CellDescription().id(3).pos({10.0f, 12.0f}),
        CellDescription()
            .id(4)
            .pos({side == Side::Left ? 9.0f : 11.0f, 11.0f})
            .angleToFront(side == Side::Left ? -90.0f : 90.0f)
            .cellType(MuscleDescription().mode(AutoBendingDescription().maxAngleDeviation(MaxAngleDeviation * 2 / 90.0f)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, getValue(channel0)).weight(1, 0, getValue(channel1) / 4)),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(4, 2);

    _simulationFacade->setSimulationData(data);

    auto minAngle = 90.0f;
    auto maxAngle = 90.0f;
    for (int i = 0; i < 200; ++i) {
        _simulationFacade->calcTimesteps(10);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = getCell(actualData, 4);
        auto actualCell1 = getCell(actualData, 1);
        auto actualCell2 = getCell(actualData, 2);
        auto actualCell3 = getCell(actualData, 3);

        ASSERT_EQ(4, actualData._cells.size());

        EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell2._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell2._connections.at(1)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));

        auto angle = actualCell2._connections.at(side == Side::Left ? 2 : 1)._angleFromPrevious;
        minAngle = std::min(minAngle, angle);
        maxAngle = std::max(maxAngle, angle);
        if (i == 0) {
            if (channel0 == Channel0::Zero) {
                EXPECT_TRUE(angle < 90.0f + NEAR_ZERO);
                EXPECT_TRUE(angle > 90.0f - NEAR_ZERO);
            } else if ((side == Side::Left && channel0 == Channel0::Positive) || (side == Side::Right && channel0 == Channel0::Negative)) {
                EXPECT_TRUE(angle > 90.0f + NEAR_ZERO);
            } else {
                EXPECT_TRUE(angle < 90.0f - NEAR_ZERO);
            }
        }
    }

    if (channel0 == Channel0::Zero) {
        EXPECT_TRUE(minAngle < 90.0f + AnglePrecision);
        EXPECT_TRUE(minAngle > 90.0f - AnglePrecision);
        EXPECT_TRUE(maxAngle < 90.0f + AnglePrecision);
        EXPECT_TRUE(maxAngle > 90.0f - AnglePrecision);
    } else if (channel1 == Channel1::Zero) {
        EXPECT_TRUE(minAngle < 90.0f - MaxAngleDeviation + AnglePrecision);
        EXPECT_TRUE(minAngle > 90.0f - MaxAngleDeviation - AnglePrecision);
        EXPECT_TRUE(maxAngle > 90.0f + MaxAngleDeviation - AnglePrecision);
        EXPECT_TRUE(maxAngle < 90.0f + MaxAngleDeviation + AnglePrecision);
    }
}

class MuscleTests_ManualBending_New
    : public MuscleTests_New
    , public testing::WithParamInterface<std::tuple<Side, Channel0>>
{
};

INSTANTIATE_TEST_SUITE_P(
    MuscleTests_ManualBending_New,
    MuscleTests_ManualBending_New,
    ::testing::Values(
        std::make_tuple(Side::Left, Channel0::Zero),
        std::make_tuple(Side::Right, Channel0::Zero),
        std::make_tuple(Side::Left, Channel0::Positive),
        std::make_tuple(Side::Right, Channel0::Positive),
        std::make_tuple(Side::Left, Channel0::Negative),
        std::make_tuple(Side::Right, Channel0::Negative)));

TEST_P(MuscleTests_ManualBending_New, muscleWithTwoConnections)
{
    auto constexpr MaxAngleDeviation = 30.0f;
    auto constexpr AnglePrecision = NEAR_ZERO;

    auto [side, channel0] = GetParam();

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({side == Side::Left ? 10.0f : 12.0f, 10.0f}).cellType(OscillatorDescription().autoTriggerInterval(20)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .angleToFront(side == Side::Left ? 90.0f : -90.0f)
            .cellType(MuscleDescription().mode(ManualBendingDescription().maxAngleDeviation(MaxAngleDeviation * 2 / 180.0f)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, getValue(channel0))),
        CellDescription().id(3).pos({side == Side::Left ? 12.0f : 10.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);

    auto minAngle = 180.0f;
    auto maxAngle = 180.0f;
    auto numPositiveAngleChanges = 0;
    auto numNegativeAngleChanges = 0;
    std::optional<float> lastAngle;
    for (int i = 0; i < 200; ++i) {
        _simulationFacade->calcTimesteps(10);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = getCell(actualData, 2);
        auto actualCell1 = getCell(actualData, 1);
        auto actualCell3 = getCell(actualData, 3);

        ASSERT_EQ(3, actualData._cells.size());

        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(1)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));

        auto angle = actualMuscleCell._connections.at(0)._angleFromPrevious;
        minAngle = std::min(minAngle, angle);
        maxAngle = std::max(maxAngle, angle);
        if (lastAngle.has_value()) {
            auto angleChange = angle - lastAngle.value();
            if (angleChange > 0) {
                ++numPositiveAngleChanges;
            } else if (angleChange < 0) {
                ++numNegativeAngleChanges;
            }
        }
        lastAngle = angle;
    }

    if (channel0 == Channel0::Positive && side == Side::Left) {
        EXPECT_TRUE(numPositiveAngleChanges > 10);
        EXPECT_EQ(0, numNegativeAngleChanges);
    }
    if (channel0 == Channel0::Positive && side == Side::Right) {
        EXPECT_EQ(0, numPositiveAngleChanges);
        EXPECT_TRUE(numNegativeAngleChanges > 10);
    }
    if (channel0 == Channel0::Negative && side == Side::Left) {
        EXPECT_EQ(0, numPositiveAngleChanges);
        EXPECT_TRUE(numNegativeAngleChanges < 10);
    }
    if (channel0 == Channel0::Negative && side == Side::Right) {
        EXPECT_TRUE(numPositiveAngleChanges < 10);
        EXPECT_EQ(0, numNegativeAngleChanges);
    }
    if (channel0 == Channel0::Zero) {
        EXPECT_TRUE(minAngle < 180.0f + AnglePrecision);
        EXPECT_TRUE(minAngle > 180.0f - AnglePrecision);
        EXPECT_TRUE(maxAngle < 180.0f + AnglePrecision);
        EXPECT_TRUE(maxAngle > 180.0f - AnglePrecision);
    } else if ((channel0 == Channel0::Positive && side == Side::Left) || (channel0 == Channel0::Negative && side == Side::Right)) {
        EXPECT_TRUE(minAngle < 180.0f + AnglePrecision);
        EXPECT_TRUE(minAngle > 180.0f - AnglePrecision);
        EXPECT_TRUE(maxAngle > 180.0f + MaxAngleDeviation - AnglePrecision);
        EXPECT_TRUE(maxAngle < 180.0f + MaxAngleDeviation + AnglePrecision);
    } else {
        EXPECT_TRUE(maxAngle < 180.0f + AnglePrecision);
        EXPECT_TRUE(maxAngle > 180.0f - AnglePrecision);
        EXPECT_TRUE(minAngle > 180.0f - MaxAngleDeviation - AnglePrecision);
        EXPECT_TRUE(minAngle < 180.0f - MaxAngleDeviation + AnglePrecision);
    }
}

TEST_P(MuscleTests_ManualBending_New, muscleWithOneConnection)
{
    auto constexpr MaxAngleDeviation = 30.0f;
    auto constexpr AnglePrecision = NEAR_ZERO;

    auto [side, channel0] = GetParam();

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}),
        CellDescription().id(2).pos({10.0f, 11.0f}).cellType(OscillatorDescription().autoTriggerInterval(20)),
        CellDescription().id(3).pos({10.0f, 12.0f}),
        CellDescription()
            .id(4)
            .pos({side == Side::Left ? 9.0f : 11.0f, 11.0f})
            .angleToFront(side == Side::Left ? -90.0f : 90.0f)
            .cellType(MuscleDescription().mode(ManualBendingDescription().maxAngleDeviation(MaxAngleDeviation * 2 / 90.0f).frontBackVelRatio(0.2f)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, getValue(channel0))),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(4, 2);

    _simulationFacade->setSimulationData(data);

    auto minAngle = 90.0f;
    auto maxAngle = 90.0f;
    auto numPositiveAngleChanges = 0;
    auto numNegativeAngleChanges = 0;
    std::optional<float> lastAngle;
    for (int i = 0; i < 200; ++i) {
        _simulationFacade->calcTimesteps(10);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = getCell(actualData, 4);
        auto actualCell1 = getCell(actualData, 1);
        auto actualCell2 = getCell(actualData, 2);
        auto actualCell3 = getCell(actualData, 3);

        ASSERT_EQ(4, actualData._cells.size());

        EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell2._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell2._connections.at(1)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));

        auto angle = actualCell2._connections.at(side == Side::Left ? 2 : 1)._angleFromPrevious;
        minAngle = std::min(minAngle, angle);
        maxAngle = std::max(maxAngle, angle);
        if (lastAngle.has_value()) {
            auto angleChange = angle - lastAngle.value();
            if (angleChange > 0) {
                ++numPositiveAngleChanges;
            } else if (angleChange < 0) {
                ++numNegativeAngleChanges;
            }
        }
        lastAngle = angle;
    }
    if (channel0 == Channel0::Positive && side == Side::Left) {
        EXPECT_TRUE(numPositiveAngleChanges > 10);
        EXPECT_EQ(0, numNegativeAngleChanges);
    }
    if (channel0 == Channel0::Positive && side == Side::Right) {
        EXPECT_EQ(0, numPositiveAngleChanges);
        EXPECT_TRUE(numNegativeAngleChanges > 10);
    }
    if (channel0 == Channel0::Negative && side == Side::Left) {
        EXPECT_EQ(0, numPositiveAngleChanges);
        EXPECT_TRUE(numNegativeAngleChanges < 10);
    }
    if (channel0 == Channel0::Negative && side == Side::Right) {
        EXPECT_TRUE(numPositiveAngleChanges < 10);
        EXPECT_EQ(0, numNegativeAngleChanges);
    }
    if (channel0 == Channel0::Zero) {
        EXPECT_TRUE(minAngle < 90.0f + AnglePrecision);
        EXPECT_TRUE(minAngle > 90.0f - AnglePrecision);
        EXPECT_TRUE(maxAngle < 90.0f + AnglePrecision);
        EXPECT_TRUE(maxAngle > 90.0f - AnglePrecision);
    } else if ((channel0 == Channel0::Positive && side == Side::Left) || (channel0 == Channel0::Negative && side == Side::Right)) {
        EXPECT_TRUE(minAngle < 90.0f + AnglePrecision);
        EXPECT_TRUE(minAngle > 90.0f - AnglePrecision);
        EXPECT_TRUE(maxAngle > 90.0f + MaxAngleDeviation - AnglePrecision);
        EXPECT_TRUE(maxAngle < 90.0f + MaxAngleDeviation + AnglePrecision);
    } else {
        EXPECT_TRUE(maxAngle < 90.0f + AnglePrecision);
        EXPECT_TRUE(maxAngle > 90.0f - AnglePrecision);
        EXPECT_TRUE(minAngle > 90.0f - MaxAngleDeviation - AnglePrecision);
        EXPECT_TRUE(minAngle < 90.0f - MaxAngleDeviation + AnglePrecision);
    }
}

class MuscleTests_AngleBending_New
    : public MuscleTests_New
    , public testing::WithParamInterface<std::tuple<Side, float>>
{};

INSTANTIATE_TEST_SUITE_P(
    MuscleTests_AngleBending_New,
    MuscleTests_AngleBending_New,
    ::testing::Values(
        std::make_tuple(Side::Left, 0.0f),
        std::make_tuple(Side::Left, -30.0f),
        std::make_tuple(Side::Left, -60.0f),
        std::make_tuple(Side::Left, -90.0f),
        std::make_tuple(Side::Left, -120.0f),
        std::make_tuple(Side::Left, -150.0f),
        std::make_tuple(Side::Left, -180.0f),
        std::make_tuple(Side::Right, 0.0f),
        std::make_tuple(Side::Right, 30.0f),
        std::make_tuple(Side::Right, 60.0f),
        std::make_tuple(Side::Right, 90.0f),
        std::make_tuple(Side::Right, 120.0f),
        std::make_tuple(Side::Right, 150.0f),
        std::make_tuple(Side::Right, 180.0f)));

TEST_P(MuscleTests_AngleBending_New, muscleWithTwoConnections)
{
    auto constexpr MaxAngleDeviation = 120.0f;
    auto constexpr AnglePrecision = 2.0f;

    auto [side, targetAngle] = GetParam();

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({side == Side::Left ? 10.0f : 12.0f, 10.0f}).cellType(OscillatorDescription().autoTriggerInterval(10)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .angleToFront(side == Side::Left ? 90.0f : -90.0f)
            .cellType(MuscleDescription().mode(AngleBendingDescription().maxAngleDeviation(MaxAngleDeviation * 2 / 180.0f)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, 1.0f).weight(1, 0, targetAngle / 180.0f)),
        CellDescription().id(3).pos({side == Side::Left ? 12.0f : 10.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualMuscleCell = getCell(actualData, 2);
    auto actualCell1 = getCell(actualData, 1);
    auto actualCell3 = getCell(actualData, 3);

    ASSERT_EQ(3, actualData._cells.size());

    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(1)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));

    auto angle = actualMuscleCell._connections.at(0)._angleFromPrevious;
    if (side == Side::Left) {
        EXPECT_TRUE(abs(270.0f - angle + targetAngle) < AnglePrecision);
    } else {
        EXPECT_TRUE(abs(angle - 90.0f - targetAngle) < AnglePrecision);
    }
}

TEST_P(MuscleTests_AngleBending_New, muscleWithOneConnection)
{
    auto constexpr MaxAngleDeviation = 120.0f;
    auto constexpr AngleMinDistance = 30.0f;
    auto constexpr AnglePrecision = 2.0f;

    auto [side, targetAngle] = GetParam();

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}),
        CellDescription().id(2).pos({10.0f, 11.0f}).cellType(OscillatorDescription().autoTriggerInterval(10)),
        CellDescription().id(3).pos({10.0f, 12.0f}),
        CellDescription()
            .id(4)
            .pos({side == Side::Left ? 9.0f : 11.0f, 11.0f})
            .angleToFront(side == Side::Left ? -90.0f : 90.0f)
            .cellType(MuscleDescription().mode(AngleBendingDescription().maxAngleDeviation(MaxAngleDeviation * 2 / 90.0f).frontBackVelRatio(0.2f)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, 1.0f).weight(1, 0, targetAngle / 180.0f)),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(4, 2);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualMuscleCell = getCell(actualData, 4);
    auto actualCell1 = getCell(actualData, 1);
    auto actualCell2 = getCell(actualData, 2);
    auto actualCell3 = getCell(actualData, 3);

    ASSERT_EQ(4, actualData._cells.size());

    EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualCell2._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualCell2._connections.at(1)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));

    auto angle = actualCell2._connections.at(side == Side::Left ? 2 : 1)._angleFromPrevious;
    if (side == Side::Left) {
        targetAngle = std::min(-AngleMinDistance, std::max(-180.0f + AngleMinDistance, targetAngle));
        EXPECT_TRUE(abs(angle - targetAngle - 180.0f) < AnglePrecision);
    } else {
        targetAngle = std::min(180.0f - AngleMinDistance, std::max(AngleMinDistance, targetAngle));
        EXPECT_TRUE(abs(angle - targetAngle) < AnglePrecision);
    }
}

class MuscleTests_AutoCrawling_New
    : public MuscleTests_New
    , public testing::WithParamInterface<Channel0>
{};

INSTANTIATE_TEST_SUITE_P(
    MuscleTests_AutoCrawling_New,
    MuscleTests_AutoCrawling_New,
    ::testing::Values(
        Channel0::Positive,
        Channel0::Negative,
        Channel0::Zero));

TEST_P(MuscleTests_AutoCrawling_New, muscleWithTwoConnections)
{
    auto constexpr MaxDistanceDeviation = 0.8f;

    auto channel0 = GetParam();

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellType(OscillatorDescription().autoTriggerInterval(10)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(MuscleDescription().mode(AutoCrawlingDescription().maxDistanceDeviation(MaxDistanceDeviation)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, getValue(channel0))),
        CellDescription().id(3).pos({12.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);

    auto minDistance = 1.0f;
    auto maxDistance = 1.0f;
    for (int i = 0; i < 200; ++i) {
        _simulationFacade->calcTimesteps(10);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = getCell(actualData, 2);
        auto actualCell1 = getCell(actualData, 1);
        auto actualCell3 = getCell(actualData, 3);

        ASSERT_EQ(3, actualData._cells.size());

        EXPECT_TRUE(approxCompare(180.0f, actualMuscleCell._connections.at(0)._angleFromPrevious));
        EXPECT_TRUE(approxCompare(180.0f, actualMuscleCell._connections.at(1)._angleFromPrevious));
        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(1)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));

        auto distance = actualMuscleCell._connections.at(0)._distance;
        minDistance = std::min(minDistance, distance);
        maxDistance = std::max(maxDistance, distance);
        if (i == 0) {
            if (channel0 == Channel0::Zero) {
                EXPECT_TRUE(distance < 1.0f + NEAR_ZERO);
                EXPECT_TRUE(distance > 1.0f - NEAR_ZERO);
            } else if (channel0 == Channel0::Positive) {
                EXPECT_TRUE(distance < 1.0f - NEAR_ZERO);
            } else {
                EXPECT_TRUE(distance > 1.0f + NEAR_ZERO);
            }
        }
    }

    if (channel0 == Channel0::Zero) {
        EXPECT_TRUE(minDistance < 1.0f + NEAR_ZERO);
        EXPECT_TRUE(minDistance > 1.0f - NEAR_ZERO);
        EXPECT_TRUE(maxDistance < 1.0f + NEAR_ZERO);
        EXPECT_TRUE(maxDistance > 1.0f - NEAR_ZERO);
    } else {
        EXPECT_TRUE(minDistance < 1.0f - MaxDistanceDeviation + NEAR_ZERO);
        EXPECT_TRUE(minDistance > 1.0f - MaxDistanceDeviation - NEAR_ZERO);
        EXPECT_TRUE(maxDistance > 1.0f + MaxDistanceDeviation - NEAR_ZERO);
        EXPECT_TRUE(maxDistance < 1.0f + MaxDistanceDeviation + NEAR_ZERO);
    }
}

TEST_P(MuscleTests_AutoCrawling_New, muscleWithOneConnection)
{
    auto constexpr MaxDistanceDeviation = 0.8f;

    auto channel0 = GetParam();

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellType(OscillatorDescription().autoTriggerInterval(10)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(MuscleDescription().mode(AutoCrawlingDescription().maxDistanceDeviation(MaxDistanceDeviation)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, getValue(channel0))),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);

    auto minDistance = 1.0f;
    auto maxDistance = 1.0f;
    for (int i = 0; i < 200; ++i) {
        _simulationFacade->calcTimesteps(10);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = getCell(actualData, 2);
        auto actualCell1 = getCell(actualData, 1);

        ASSERT_EQ(2, actualData._cells.size());

        auto distance = actualMuscleCell._connections.at(0)._distance;
        EXPECT_TRUE(approxCompare(distance, actualCell1._connections.at(0)._distance));

        minDistance = std::min(minDistance, distance);
        maxDistance = std::max(maxDistance, distance);
        if (i == 0) {
            if (channel0 == Channel0::Zero) {
                EXPECT_TRUE(distance < 1.0f + NEAR_ZERO);
                EXPECT_TRUE(distance > 1.0f - NEAR_ZERO);
            } else if (channel0 == Channel0::Positive) {
                EXPECT_TRUE(distance < 1.0f - NEAR_ZERO);
            } else {
                EXPECT_TRUE(distance > 1.0f + NEAR_ZERO);
            }
        }
    }

    if (channel0 == Channel0::Zero) {
        EXPECT_TRUE(minDistance < 1.0f + NEAR_ZERO);
        EXPECT_TRUE(minDistance > 1.0f - NEAR_ZERO);
        EXPECT_TRUE(maxDistance < 1.0f + NEAR_ZERO);
        EXPECT_TRUE(maxDistance > 1.0f - NEAR_ZERO);
    } else {
        EXPECT_TRUE(minDistance < 1.0f - MaxDistanceDeviation + NEAR_ZERO);
        EXPECT_TRUE(minDistance > 1.0f - MaxDistanceDeviation - NEAR_ZERO);
        EXPECT_TRUE(maxDistance > 1.0f + MaxDistanceDeviation - NEAR_ZERO);
        EXPECT_TRUE(maxDistance < 1.0f + MaxDistanceDeviation + NEAR_ZERO);
    }
}

class MuscleTests_ManualCrawling_New
    : public MuscleTests_New
    , public testing::WithParamInterface<Channel0>
{};

INSTANTIATE_TEST_SUITE_P(
    MuscleTests_ManualCrawling_New,
    MuscleTests_ManualCrawling_New,
    ::testing::Values(Channel0::Positive, Channel0::Negative, Channel0::Zero));

TEST_P(MuscleTests_ManualCrawling_New, muscleWithTwoConnections)
{
    auto constexpr MaxDistanceDeviation = 0.8f;

    auto channel0 = GetParam();

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellType(OscillatorDescription().autoTriggerInterval(10)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(MuscleDescription().mode(ManualCrawlingDescription().maxDistanceDeviation(MaxDistanceDeviation)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, getValue(channel0))),
        CellDescription().id(3).pos({12.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);

    auto minDistance = 1.0f;
    auto maxDistance = 1.0f;
    for (int i = 0; i < 200; ++i) {
        _simulationFacade->calcTimesteps(10);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = getCell(actualData, 2);
        auto actualCell1 = getCell(actualData, 1);

        ASSERT_EQ(3, actualData._cells.size());

        auto distance = actualMuscleCell._connections.at(0)._distance;
        EXPECT_TRUE(approxCompare(distance, actualCell1._connections.at(0)._distance));

        minDistance = std::min(minDistance, distance);
        maxDistance = std::max(maxDistance, distance);
    }

    if (channel0 == Channel0::Zero) {
        EXPECT_TRUE(approxCompare(1.0f, minDistance));
        EXPECT_TRUE(approxCompare(1.0f, maxDistance));
    } else if (channel0 == Channel0::Positive) {
        EXPECT_TRUE(approxCompare(1.0f - MaxDistanceDeviation, minDistance));
        EXPECT_TRUE(approxCompare(1.0f, maxDistance));
    } else if (channel0 == Channel0::Negative) {
        EXPECT_TRUE(approxCompare(1.0f, minDistance));
        EXPECT_TRUE(approxCompare(1.0f + MaxDistanceDeviation, maxDistance));
    }
}

TEST_P(MuscleTests_ManualCrawling_New, muscleWithOneConnection)
{
    auto constexpr MaxDistanceDeviation = 0.8f;

    auto channel0 = GetParam();

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellType(OscillatorDescription().autoTriggerInterval(10)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(MuscleDescription().mode(ManualCrawlingDescription().maxDistanceDeviation(MaxDistanceDeviation)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, getValue(channel0))),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);

    auto minDistance = 1.0f;
    auto maxDistance = 1.0f;
    for (int i = 0; i < 200; ++i) {
        _simulationFacade->calcTimesteps(10);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = getCell(actualData, 2);
        auto actualCell1 = getCell(actualData, 1);

        ASSERT_EQ(2, actualData._cells.size());

        auto distance = actualMuscleCell._connections.at(0)._distance;
        EXPECT_TRUE(approxCompare(distance, actualCell1._connections.at(0)._distance));

        minDistance = std::min(minDistance, distance);
        maxDistance = std::max(maxDistance, distance);
    }

    if (channel0 == Channel0::Zero) {
        EXPECT_TRUE(approxCompare(1.0f, minDistance));
        EXPECT_TRUE(approxCompare(1.0f, maxDistance));
    } else if (channel0 == Channel0::Positive) {
        EXPECT_TRUE(approxCompare(1.0f - MaxDistanceDeviation, minDistance));
        EXPECT_TRUE(approxCompare(1.0f, maxDistance));
    } else if (channel0 == Channel0::Negative) {
        EXPECT_TRUE(approxCompare(1.0f, minDistance));
        EXPECT_TRUE(approxCompare(1.0f + MaxDistanceDeviation, maxDistance));
    }
}

class MuscleTests_DirectMovement_New
    : public MuscleTests_New
    , public testing::WithParamInterface<std::tuple<Channel0, Channel1>>
{};

INSTANTIATE_TEST_SUITE_P(
    MuscleTests_DirectMovement_New,
    MuscleTests_DirectMovement_New,
    ::testing::Values(
        std::make_tuple(Channel0::Positive, Channel1::Zero),
        std::make_tuple(Channel0::Negative, Channel1::Zero),
        std::make_tuple(Channel0::Zero, Channel1::Zero),
        std::make_tuple(Channel0::Positive, Channel1::Positive),
        std::make_tuple(Channel0::Negative, Channel1::Positive),
        std::make_tuple(Channel0::Zero, Channel1::Positive),
        std::make_tuple(Channel0::Positive, Channel1::Negative),
        std::make_tuple(Channel0::Negative, Channel1::Negative),
        std::make_tuple(Channel0::Zero, Channel1::Negative)));

TEST_P(MuscleTests_DirectMovement_New, muscleWithTwoConnections)
{
    auto constexpr AnglePrecision = 1.0f;
    auto [channel0, channel1] = GetParam();

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellType(OscillatorDescription().autoTriggerInterval(3)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(MuscleDescription().mode(DirectMovementDescription()))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, getValue(channel0)).weight(1, 0, getValue(channel1) / 2)),
        CellDescription().id(3).pos({12.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(3);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualCell1 = getCell(actualData, 1);
    auto actualCell2 = getCell(actualData, 2);
    auto actualCell3 = getCell(actualData, 3);

    ASSERT_EQ(1, actualCell1._connections.size());
    ASSERT_EQ(2, actualCell2._connections.size());
    ASSERT_EQ(1, actualCell3._connections.size());

    EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualCell2._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualCell2._connections.at(1)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(180.0f, actualCell2._connections.at(0)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, actualCell2._connections.at(1)._angleFromPrevious));

    auto angleVel = Math::angleOfVector(actualCell2._vel);
    if (channel1 == Channel1::Zero) {
        if (channel0 == Channel0::Positive) {
            EXPECT_TRUE(approxCompare(270.0f, angleVel, AnglePrecision));
        } else if (channel0 == Channel0::Negative) {
            EXPECT_TRUE(approxCompare(90.0f, angleVel, AnglePrecision));
        } else {
            EXPECT_TRUE(approxCompare(0.0f, Math::length(actualCell2._vel), 0.01f));
        }
    } else if (channel1 == Channel1::Positive) {
        if (channel0 == Channel0::Positive) {
            EXPECT_TRUE(approxCompare(0.0f, Math::normalizedAngle(angleVel, -180.0f), AnglePrecision));
        } else if (channel0 == Channel0::Negative) {
            EXPECT_TRUE(approxCompare(180.0f, angleVel, AnglePrecision));
        } else {
            EXPECT_TRUE(approxCompare(0.0f, Math::length(actualCell2._vel), 0.01f));
        }
    } else {
        if (channel0 == Channel0::Positive) {
            EXPECT_TRUE(approxCompare(180.0f, angleVel, AnglePrecision));
        } else if (channel0 == Channel0::Negative) {
            EXPECT_TRUE(approxCompare(0.0f, Math::normalizedAngle(angleVel, -180.0f), AnglePrecision));
        } else {
            EXPECT_TRUE(approxCompare(0.0f, Math::length(actualCell2._vel), 0.01f));
        }
    }
}
