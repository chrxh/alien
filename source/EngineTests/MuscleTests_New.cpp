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
    MuscleTests_BothSides_New,
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

TEST_P(MuscleTests_AutoBending_New, autobending)
{
    auto constexpr MaxAngleDeviation = 30.0f;
    auto constexpr AnglePrecision = NEAR_ZERO;

    auto [side, channel0, channel1] = GetParam();
    auto isParameter = [&](Side s, Channel0 c0, Channel1 c1) { return side == s && channel0 == c0 && channel1 == c1; };

    DataDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellType(OscillatorDescription().autoTriggerInterval(20)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .absAngleToConnection0(side == Side::Left ? -90.0f : 90.0f)
            .cellType(MuscleDescription().mode(AutoBendingDescription().maxAngleDeviation(MaxAngleDeviation / 180.0f)))
            .neuralNetwork(NeuralNetworkDescription().weight(0, 0, getValue(channel0)).weight(1, 0, getValue(channel1))),
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
            if (channel0 == Channel0::Zero) {
                EXPECT_TRUE(actualMuscleCell._connections.at(0)._angleFromPrevious < 180.0f + NEAR_ZERO);
                EXPECT_TRUE(actualMuscleCell._connections.at(0)._angleFromPrevious > 180.0f - NEAR_ZERO);
            } else if (
                isParameter(Side::Left, Channel0::Positive, Channel1::Zero)
                || isParameter(Side::Right, Channel0::Negative, Channel1::Zero)
                || isParameter(Side::Left, Channel0::Positive, Channel1::Positive)
                || isParameter(Side::Right, Channel0::Negative, Channel1::Positive)
                || isParameter(Side::Left, Channel0::Negative, Channel1::Negative)
                || isParameter(Side::Right, Channel0::Positive, Channel1::Negative)) {
                EXPECT_TRUE(actualMuscleCell._connections.at(0)._angleFromPrevious > 180.0f + NEAR_ZERO);
            } else {
                EXPECT_TRUE(actualMuscleCell._connections.at(0)._angleFromPrevious < 180.0f - NEAR_ZERO);
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
    } else if (isParameter(Side::Left, Channel0::Positive, Channel1::Positive)
        || isParameter(Side::Left, Channel0::Negative, Channel1::Negative)
        || isParameter(Side::Right, Channel0::Positive, Channel1::Negative)
        || isParameter(Side::Right, Channel0::Negative, Channel1::Positive)) {
        EXPECT_TRUE(minAngle < 180.0f + MaxAngleDeviation + AnglePrecision);
        EXPECT_TRUE(minAngle > 180.0f - NEAR_ZERO);
        EXPECT_TRUE(maxAngle < 180.0f + MaxAngleDeviation + AnglePrecision);
        EXPECT_TRUE(maxAngle > 180.0f + MaxAngleDeviation - AnglePrecision);
    } else {
        EXPECT_TRUE(minAngle < 180.0f - MaxAngleDeviation + AnglePrecision);
        EXPECT_TRUE(minAngle > 180.0f - MaxAngleDeviation - AnglePrecision);
        EXPECT_TRUE(maxAngle > 180.0f - MaxAngleDeviation + AnglePrecision);
        EXPECT_TRUE(maxAngle < 180.0f + NEAR_ZERO);
    }
}
