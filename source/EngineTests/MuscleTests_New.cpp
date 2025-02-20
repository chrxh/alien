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

TEST_P(MuscleTests_AutoBending_New, numConnectionsEquals2)
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
            .absAngleToConnection0(side == Side::Left ? -90.0f : 90.0f)
            .cellType(MuscleDescription().mode(AutoBendingDescription().maxAngleDeviation(MaxAngleDeviation * 2 / 180.0f)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, getValue(channel0)).weight(1, 0, getValue(channel1) / 4)),
        CellDescription().id(3).pos({side == Side::Left ? 12.0f : 10.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);

    auto minAngle = 180.0f;
    auto maxAngle = 180.0f;
    auto sumAngleChanges = 0.0f;
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
            sumAngleChanges += std::abs(angle - lastAngle.value());
        }
        lastAngle = angle;
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
        if ((side == Side::Right && channel1 == Channel1::Positive) || (side == Side::Left && channel1 == Channel1::Negative)) {
            EXPECT_TRUE(sumAngleChanges < 100.0f);
        } else {
            EXPECT_TRUE(sumAngleChanges > 100.0f);
        }
    }
}

TEST_P(MuscleTests_AutoBending_New, numConnectionsEquals1)
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
            .absAngleToConnection0(side == Side::Left ? -90.0f : 90.0f)
            .cellType(MuscleDescription().mode(AutoBendingDescription().maxAngleDeviation(MaxAngleDeviation * 2 / 90.0f)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, getValue(channel0)).weight(1, 0, getValue(channel1) / 4)),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(4, 2);

    _simulationFacade->setSimulationData(data);

    auto minAngle = 90.0f;
    auto maxAngle = 90.0f;
    auto sumAngleChanges = 0.0f;
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
            sumAngleChanges += std::abs(angle - lastAngle.value());
        }
        lastAngle = angle;
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
        if ((side == Side::Right && channel1 == Channel1::Positive) || (side == Side::Left && channel1 == Channel1::Negative)) {
            EXPECT_TRUE(sumAngleChanges < 100.0f);
        } else {
            EXPECT_TRUE(sumAngleChanges > 100.0f);
        }
    }
}

class MuscleTests_ManualBending_New
    : public MuscleTests_New
    , public testing::WithParamInterface<std::tuple<Side, Channel0>>
{
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

TEST_P(MuscleTests_ManualBending_New, numConnectionsEquals2)
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
            .absAngleToConnection0(side == Side::Left ? -90.0f : 90.0f)
            .cellType(MuscleDescription().mode(ManualBendingDescription().maxAngleDeviation(MaxAngleDeviation * 2 / 180.0f)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, getValue(channel0))),
        CellDescription().id(3).pos({side == Side::Left ? 12.0f : 10.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);

    auto minAngle = 180.0f;
    auto maxAngle = 180.0f;
    auto sumAngleChanges = 0.0f;
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
            sumAngleChanges += std::abs(angle - lastAngle.value());
        }
        lastAngle = angle;
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

TEST_P(MuscleTests_ManualBending_New, numConnectionsEquals1)
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
            .absAngleToConnection0(side == Side::Left ? -90.0f : 90.0f)
            .cellType(MuscleDescription().mode(ManualBendingDescription().maxAngleDeviation(MaxAngleDeviation * 2 / 90.0f)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, getValue(channel0))),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(4, 2);

    _simulationFacade->setSimulationData(data);

    auto minAngle = 90.0f;
    auto maxAngle = 90.0f;
    auto sumAngleChanges = 0.0f;
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
            sumAngleChanges += std::abs(angle - lastAngle.value());
        }
        lastAngle = angle;
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
