#include <cmath>

#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

enum class DetailedPreview
{
    No,
    Yes
};

class MuscleTests : public IntegrationTestFramework
{
public:
    MuscleTests()
        : IntegrationTestFramework()
    {}

    ~MuscleTests() = default;

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

    void setSimulationData(Desc const& data, DetailedPreview detailedPreview)
    {
        if (detailedPreview == DetailedPreview::No) {
            _simulationFacade->setSimulationData(data);
        } else {
            _simulationFacade->setPreviewData(data);
        }
    }

    void calcTimesteps(int timesteps, DetailedPreview detailedPreview)
    {
        if (detailedPreview == DetailedPreview::No) {
            _simulationFacade->calcTimesteps(timesteps);
        } else {
            _simulationFacade->calcTimestepsForPreview(timesteps, true);
        }
    }

    Desc getSimulationData(DetailedPreview detailedPreview)
    {
        if (detailedPreview == DetailedPreview::No) {
            return _simulationFacade->getSimulationData();
        } else {
            return _simulationFacade->getPreviewData();
        }
    }
};

TEST_F(MuscleTests, noFrontAngle)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}),
            ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().neuralNetwork(NeuralNetDesc().bias(0, 1.0f)).cellType(MuscleDesc().mode(AutoBendingDesc()))),
            ObjectDesc().id(3).pos({12.0f, 10.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualMuscleCell = actualData.getObjectRef(2);

    ASSERT_EQ(3, actualData._objects.size());
    EXPECT_TRUE(approxCompare(180.0f, actualMuscleCell._connections.at(0)._angleFromPrevious));
}


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

class MuscleTests_AutoBending
    : public MuscleTests
    , public testing::WithParamInterface<std::tuple<Side, Channel0, Channel1>>
{};

INSTANTIATE_TEST_SUITE_P(
    MuscleTests_AutoBending,
    MuscleTests_AutoBending,
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

TEST_P(MuscleTests_AutoBending, muscleWithTwoConnections)
{
    auto constexpr MaxAngleDeviation = 15.0f;
    auto constexpr AnglePrecision = NEAR_ZERO;

    auto [side, channel0, channel1] = GetParam();

    NeuralNetDesc nn;
    nn._weights.clear();
    nn._weights.resize(NEURONS_PER_CELL * NEURONS_PER_CELL, NeuralNetWeight(0));
    nn._biases.at(Channels::CellTypeActivation) = getValue(channel0);
    nn._biases.at(Channels::MuscleAngle) = getValue(channel1) / 4;

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({side == Side::Left ? 10.0f : 13.0f, 10.0f}).type(CellDesc()),
            ObjectDesc()
                .id(2)
                .pos({side == Side::Left ? 11.0f : 12.0f, 10.0f})
                .type(CellDesc()
                          .frontAngle(side == Side::Left ? -90.0f : 90.0f)
                          .headCell(true)
                          .cellType(MuscleDesc().mode(AutoBendingDesc().maxAngleDeviation(MaxAngleDeviation * 2 / 180.0f)))
                          .neuralNetwork(nn)),
            ObjectDesc().id(3).pos({side == Side::Left ? 12.0f : 11.0f, 10.0f}).type(CellDesc()),
            ObjectDesc().id(4).pos({side == Side::Left ? 13.0f : 10.0f, 10.0f}).type(CellDesc()),
        },
        CreatureDesc().id(0),
        GenomeDesc().frontAngle(side == Side::Left ? -90.0f : 90.0f));
    data.addConnection(3, 4);
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);

    auto minAngle = 180.0f;
    auto maxAngle = 180.0f;
    for (int i = 0; i < 250; ++i) {
        _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

        auto actualData = _simulationFacade->getSimulationData();
        ASSERT_EQ(4, actualData._objects.size());

        auto actualCell1 = actualData.getObjectRef(1);
        auto actualMuscleCell = actualData.getObjectRef(2);
        auto actualCell3 = actualData.getObjectRef(3);
        auto actualCell4 = actualData.getObjectRef(4);


        EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(1)._distance));
        EXPECT_TRUE(approxCompare(180.0f, actualMuscleCell._connections.at(0)._angleFromPrevious));
        EXPECT_TRUE(approxCompare(180.0f, actualMuscleCell._connections.at(1)._angleFromPrevious));
        EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell4._connections.at(0)._distance));

        auto angle = actualCell3._connections.at(1)._angleFromPrevious;
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

TEST_P(MuscleTests_AutoBending, muscleWithOneConnection)
{
    auto constexpr MaxAngleDeviation = 15.0f;
    auto constexpr AnglePrecision = NEAR_ZERO;

    auto [side, channel0, channel1] = GetParam();

    NeuralNetDesc nn;
    nn._weights.clear();
    nn._weights.resize(NEURONS_PER_CELL * NEURONS_PER_CELL, NeuralNetWeight(0));
    nn._biases.at(Channels::CellTypeActivation) = getValue(channel0);
    nn._biases.at(Channels::MuscleAngle) = getValue(channel1) / 4;

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc()),
            ObjectDesc().id(2).pos({10.0f, 11.0f}).type(CellDesc()),
            ObjectDesc().id(3).pos({10.0f, 12.0f}).type(CellDesc()),
            ObjectDesc()
                .id(4)
                .pos({side == Side::Left ? 9.0f : 11.0f, 11.0f})
                .type(CellDesc()
                          .frontAngle(side == Side::Left ? 90.0f : -90.0f)
                          .headCell(true)
                          .cellType(MuscleDesc().mode(AutoBendingDesc().maxAngleDeviation(MaxAngleDeviation * 2 / 90.0f)))
                          .neuralNetwork(nn)),
        },
        CreatureDesc().id(0),
        GenomeDesc().frontAngle(side == Side::Left ? 90.0f : -90.0f));
    data.addConnection(2, 3);
    data.addConnection(1, 2);
    data.addConnection(4, 2);

    _simulationFacade->setSimulationData(data);

    auto minAngle = 90.0f;
    auto maxAngle = 90.0f;
    for (int i = 0; i < 250; ++i) {
        _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = actualData.getObjectRef(4);
        auto actualCell1 = actualData.getObjectRef(1);
        auto actualCell2 = actualData.getObjectRef(2);
        auto actualCell3 = actualData.getObjectRef(3);

        ASSERT_EQ(4, actualData._objects.size());

        EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell2._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell2._connections.at(1)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));

        auto angle = actualCell2._connections.at(side == Side::Left ? 1 : 2)._angleFromPrevious;
        minAngle = std::min(minAngle, angle);
        maxAngle = std::max(maxAngle, angle);
        if (i == 0) {
            if (channel0 == Channel0::Zero) {
                EXPECT_TRUE(angle < 90.0f + NEAR_ZERO);
                EXPECT_TRUE(angle > 90.0f - NEAR_ZERO);
            } else if ((side == Side::Left && channel0 == Channel0::Positive) || (side == Side::Right && channel0 == Channel0::Negative)) {
                EXPECT_TRUE(angle < 90.0f - NEAR_ZERO);
            } else {
                EXPECT_TRUE(angle > 90.0f + NEAR_ZERO);
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

class MuscleTests_ManualBending
    : public MuscleTests
    , public testing::WithParamInterface<std::tuple<Side, Channel0, DetailedPreview>>
{};

INSTANTIATE_TEST_SUITE_P(
    MuscleTests_ManualBending,
    MuscleTests_ManualBending,
    ::testing::Values(
        std::make_tuple(Side::Left, Channel0::Zero, DetailedPreview::No),
        std::make_tuple(Side::Right, Channel0::Zero, DetailedPreview::No),
        std::make_tuple(Side::Left, Channel0::Positive, DetailedPreview::No),
        std::make_tuple(Side::Right, Channel0::Positive, DetailedPreview::No),
        std::make_tuple(Side::Left, Channel0::Negative, DetailedPreview::No),
        std::make_tuple(Side::Right, Channel0::Negative, DetailedPreview::No),
        std::make_tuple(Side::Left, Channel0::Positive, DetailedPreview::Yes)));

TEST_P(MuscleTests_ManualBending, muscleWithTwoConnections)
{
    auto constexpr MaxAngleDeviation = 15.0f;
    auto constexpr AnglePrecision = NEAR_ZERO;

    auto [side, channel0, detailedPreview] = GetParam();

    NeuralNetDesc nn;
    nn._weights.clear();
    nn._weights.resize(NEURONS_PER_CELL * NEURONS_PER_CELL, NeuralNetWeight(0));
    nn._biases.at(Channels::CellTypeActivation) = getValue(channel0);

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({side == Side::Left ? 10.0f : 13.0f, 10.0f}).type(CellDesc()),
            ObjectDesc()
                .id(2)
                .pos({side == Side::Left ? 11.0f : 12.0f, 10.0f})
                .type(CellDesc()
                          .frontAngle(side == Side::Left ? -90.0f : 90.0f)
                          .headCell(true)
                          .cellType(MuscleDesc().mode(ManualBendingDesc().maxAngleDeviation(MaxAngleDeviation * 2 / 180.0f)))
                          .neuralNetwork(nn)),
            ObjectDesc().id(3).pos({side == Side::Left ? 12.0f : 11.0f, 10.0f}).type(CellDesc()),
            ObjectDesc().id(4).pos({side == Side::Left ? 13.0f : 10.0f, 10.0f}).type(CellDesc()),
        },
        CreatureDesc().id(0),
        GenomeDesc().frontAngle(side == Side::Left ? -90.0f : 90.0f));
    data.addConnection(3, 4);
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    setSimulationData(data, detailedPreview);

    auto minAngle = 180.0f;
    auto maxAngle = 180.0f;
    auto numPositiveAngleChanges = 0;
    auto numNegativeAngleChanges = 0;
    std::optional<float> lastAngle;
    for (int i = 0; i < 500; ++i) {
        calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION, detailedPreview);

        auto actualData = getSimulationData(detailedPreview);
        ASSERT_EQ(4, actualData._objects.size());

        auto actualCell1 = actualData.getObjectRef(1);
        auto actualMuscleCell = actualData.getObjectRef(2);
        auto actualCell3 = actualData.getObjectRef(3);
        auto actualCell4 = actualData.getObjectRef(4);

        EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(1)._distance));
        EXPECT_TRUE(approxCompare(180.0f, actualMuscleCell._connections.at(0)._angleFromPrevious));
        EXPECT_TRUE(approxCompare(180.0f, actualMuscleCell._connections.at(1)._angleFromPrevious));
        EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell4._connections.at(0)._distance));

        auto angle = actualCell3._connections.at(0)._angleFromPrevious;
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

    if ((channel0 == Channel0::Positive && side == Side::Left) || (channel0 == Channel0::Negative && side == Side::Right)) {
        EXPECT_EQ(0, numPositiveAngleChanges);
        EXPECT_TRUE(numNegativeAngleChanges > 10);
    }
    if ((channel0 == Channel0::Positive && side == Side::Right) || (channel0 == Channel0::Negative && side == Side::Left)) {
        EXPECT_TRUE(numPositiveAngleChanges > 10);
        EXPECT_EQ(0, numNegativeAngleChanges);
    }
    if (channel0 == Channel0::Zero) {
        EXPECT_TRUE(minAngle < 180.0f + AnglePrecision);
        EXPECT_TRUE(minAngle > 180.0f - AnglePrecision);
        EXPECT_TRUE(maxAngle < 180.0f + AnglePrecision);
        EXPECT_TRUE(maxAngle > 180.0f - AnglePrecision);
    } else if ((channel0 == Channel0::Positive && side == Side::Left) || (channel0 == Channel0::Negative && side == Side::Right)) {
        EXPECT_TRUE(maxAngle < 180.0f + AnglePrecision);
        EXPECT_TRUE(maxAngle > 180.0f - AnglePrecision);
        EXPECT_TRUE(minAngle > 180.0f - MaxAngleDeviation - AnglePrecision);
        EXPECT_TRUE(minAngle < 180.0f - MaxAngleDeviation + AnglePrecision);
    } else {
        EXPECT_TRUE(minAngle < 180.0f + AnglePrecision);
        EXPECT_TRUE(minAngle > 180.0f - AnglePrecision);
        EXPECT_TRUE(maxAngle > 180.0f + MaxAngleDeviation - AnglePrecision);
        EXPECT_TRUE(maxAngle < 180.0f + MaxAngleDeviation + AnglePrecision);
    }
}

TEST_P(MuscleTests_ManualBending, muscleWithOneConnection)
{
    auto constexpr MaxAngleDeviation = 15.0f;
    auto constexpr AnglePrecision = NEAR_ZERO;

    auto [side, channel0, detailedPreview] = GetParam();

    NeuralNetDesc nn;
    nn._weights.clear();
    nn._weights.resize(NEURONS_PER_CELL * NEURONS_PER_CELL, NeuralNetWeight(0));
    nn._biases.at(Channels::CellTypeActivation) = getValue(channel0);

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc()),
            ObjectDesc().id(2).pos({10.0f, 11.0f}).type(CellDesc()),
            ObjectDesc().id(3).pos({10.0f, 12.0f}).type(CellDesc()),
            ObjectDesc()
                .id(4)
                .pos({side == Side::Left ? 9.0f : 11.0f, 11.0f})
                .type(CellDesc()
                          .frontAngle(side == Side::Left ? 90.0f : -90.0f)
                          .headCell(true)
                          .cellType(MuscleDesc().mode(ManualBendingDesc().maxAngleDeviation(MaxAngleDeviation * 2 / 90.0f).forwardBackwardRatio(0.8f)))
                          .neuralNetwork(nn)),
        },
        CreatureDesc().id(0),
        GenomeDesc().frontAngle(side == Side::Left ? 90.0f : -90.0f));
    data.addConnection(2, 3);
    data.addConnection(1, 2);
    data.addConnection(4, 2);

    _simulationFacade->setSimulationData(data);

    auto minAngle = 90.0f;
    auto maxAngle = 90.0f;
    auto numPositiveAngleChanges = 0;
    auto numNegativeAngleChanges = 0;
    std::optional<float> lastAngle;
    for (int i = 0; i < 500; ++i) {
        _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = actualData.getObjectRef(4);
        auto actualCell1 = actualData.getObjectRef(1);
        auto actualCell2 = actualData.getObjectRef(2);
        auto actualCell3 = actualData.getObjectRef(3);

        ASSERT_EQ(4, actualData._objects.size());

        EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell2._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell2._connections.at(1)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));
        EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));

        auto angle = actualCell2._connections.at(side == Side::Left ? 1 : 2)._angleFromPrevious;
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
    if ((channel0 == Channel0::Positive && side == Side::Left) || (channel0 == Channel0::Negative && side == Side::Right)) {
        EXPECT_EQ(0, numPositiveAngleChanges);
        EXPECT_TRUE(numNegativeAngleChanges > 10);
    }
    if ((channel0 == Channel0::Positive && side == Side::Right) || (channel0 == Channel0::Negative && side == Side::Left)) {
        EXPECT_TRUE(numPositiveAngleChanges > 10);
        EXPECT_EQ(0, numNegativeAngleChanges);
    }
    if (channel0 == Channel0::Zero) {
        EXPECT_TRUE(minAngle < 90.0f + AnglePrecision);
        EXPECT_TRUE(minAngle > 90.0f - AnglePrecision);
        EXPECT_TRUE(maxAngle < 90.0f + AnglePrecision);
        EXPECT_TRUE(maxAngle > 90.0f - AnglePrecision);
    } else if ((channel0 == Channel0::Positive && side == Side::Left) || (channel0 == Channel0::Negative && side == Side::Right)) {
        EXPECT_TRUE(maxAngle < 90.0f + AnglePrecision);
        EXPECT_TRUE(maxAngle > 90.0f - AnglePrecision);
        EXPECT_TRUE(minAngle > 90.0f - MaxAngleDeviation - AnglePrecision);
        EXPECT_TRUE(minAngle < 90.0f - MaxAngleDeviation + AnglePrecision);
    } else {
        EXPECT_TRUE(minAngle < 90.0f + AnglePrecision);
        EXPECT_TRUE(minAngle > 90.0f - AnglePrecision);
        EXPECT_TRUE(maxAngle > 90.0f + MaxAngleDeviation - AnglePrecision);
        EXPECT_TRUE(maxAngle < 90.0f + MaxAngleDeviation + AnglePrecision);
    }
}

class MuscleTests_AngleBending
    : public MuscleTests
    , public testing::WithParamInterface<std::tuple<Side, float>>
{};

INSTANTIATE_TEST_SUITE_P(
    MuscleTests_AngleBending,
    MuscleTests_AngleBending,
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

TEST_P(MuscleTests_AngleBending, muscleWithTwoConnections)
{
    auto constexpr MaxAngleDeviation = 120.0f;
    auto constexpr AnglePrecision = 2.0f;

    auto [side, targetAngle] = GetParam();

    NeuralNetDesc nn;
    nn._weights.clear();
    nn._weights.resize(NEURONS_PER_CELL * NEURONS_PER_CELL, NeuralNetWeight(0));
    nn._biases.at(Channels::CellTypeActivation) = 1.0f;
    nn._biases.at(Channels::MuscleAngle) = targetAngle / 180.0f;

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({side == Side::Left ? 10.0f : 13.0f, 10.0f}).type(CellDesc()),
            ObjectDesc()
                .id(2)
                .pos({side == Side::Left ? 11.0f : 12.0f, 10.0f})
                .type(CellDesc()
                          .frontAngle(side == Side::Left ? -90.0f : 90.0f)
                          .headCell(true)
                          .cellType(MuscleDesc().mode(AngleBendingDesc().maxAngleDeviation(MaxAngleDeviation * 2 / 180.0f)))
                          .neuralNetwork(nn)),
            ObjectDesc().id(3).pos({side == Side::Left ? 12.0f : 11.0f, 10.0f}).type(CellDesc()),
            ObjectDesc().id(4).pos({side == Side::Left ? 13.0f : 10.0f, 10.0f}).type(CellDesc()),
        },
        CreatureDesc().id(0),
        GenomeDesc().frontAngle(side == Side::Left ? -90.0f : 90.0f));
    data.addConnection(3, 4);
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(500);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData._objects.size());
    auto actualCell1 = actualData.getObjectRef(1);
    auto actualMuscleCell = actualData.getObjectRef(2);
    auto actualCell3 = actualData.getObjectRef(3);
    auto actualCell4 = actualData.getObjectRef(4);

    EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(1)._distance));
    EXPECT_TRUE(approxCompare(180.0f, actualMuscleCell._connections.at(0)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, actualMuscleCell._connections.at(1)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(1.0f, actualCell4._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualCell4._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualCell4._connections.at(0)._distance));

    auto angle = actualCell3._connections.at(0)._angleFromPrevious;
    if (side == Side::Left) {
        EXPECT_TRUE(approxCompareAngles(-90.0f + targetAngle, angle, AnglePrecision));
    } else {
        EXPECT_TRUE(approxCompareAngles(targetAngle - 90.0f, angle, AnglePrecision));
    }
}

TEST_P(MuscleTests_AngleBending, muscleWithOneConnection)
{
    auto constexpr MaxAngleDeviation = 120.0f;
    auto constexpr AnglePrecision = 2.0f;

    auto [side, targetAngle] = GetParam();

    NeuralNetDesc nn;
    nn._weights.clear();
    nn._weights.resize(NEURONS_PER_CELL * NEURONS_PER_CELL, NeuralNetWeight(0));
    nn._biases.at(Channels::CellTypeActivation) = 1.0f;
    nn._biases.at(Channels::MuscleAngle) = targetAngle / 180.0f;

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc()),
            ObjectDesc().id(2).pos({10.0f, 11.0f}).type(CellDesc()),
            ObjectDesc().id(3).pos({10.0f, 12.0f}).type(CellDesc()),
            ObjectDesc()
                .id(4)
                .pos({side == Side::Left ? 9.0f : 11.0f, 11.0f})
                .type(CellDesc()
                          .frontAngle(side == Side::Left ? 90.0f : -90.0f)
                          .headCell(true)
                          .cellType(MuscleDesc().mode(AngleBendingDesc().maxAngleDeviation(MaxAngleDeviation * 2 / 90.0f).attractionRepulsionRatio(0.8f)))
                          .neuralNetwork(nn)),
        },
        CreatureDesc().id(0),
        GenomeDesc().frontAngle(side == Side::Left ? 90.0f : -90.0f));
    data.addConnection(2, 3);
    data.addConnection(1, 2);
    data.addConnection(4, 2);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(500);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualMuscleCell = actualData.getObjectRef(4);
    auto actualCell1 = actualData.getObjectRef(1);
    auto actualCell2 = actualData.getObjectRef(2);
    auto actualCell3 = actualData.getObjectRef(3);

    ASSERT_EQ(4, actualData._objects.size());

    EXPECT_TRUE(approxCompare(1.0f, actualCell1._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualCell2._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualCell2._connections.at(1)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualCell3._connections.at(0)._distance));
    EXPECT_TRUE(approxCompare(1.0f, actualMuscleCell._connections.at(0)._distance));

    auto angle = actualCell2._connections.at(side == Side::Left ? 2 : 1)._angleFromPrevious;
    if (side == Side::Left) {
        EXPECT_TRUE(approxCompareAngles(-90.0f + targetAngle, angle, AnglePrecision));
    } else {
        EXPECT_TRUE(approxCompareAngles(targetAngle - 90.0f, angle, AnglePrecision));
    }
}

class MuscleTests_AutoCrawling
    : public MuscleTests
    , public testing::WithParamInterface<Channel0>
{};

INSTANTIATE_TEST_SUITE_P(MuscleTests_AutoCrawling, MuscleTests_AutoCrawling, ::testing::Values(Channel0::Positive, Channel0::Negative, Channel0::Zero));

TEST_P(MuscleTests_AutoCrawling, muscleWithTwoConnections)
{
    auto constexpr MaxDistanceDeviation = 0.5f;

    auto channel0 = GetParam();

    NeuralNetDesc nn;
    nn._weights.clear();
    nn._weights.resize(NEURONS_PER_CELL * NEURONS_PER_CELL, NeuralNetWeight(0));
    nn._biases.at(Channels::CellTypeActivation) = getValue(channel0);

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().frontAngle(0.0f).headCell(true)),
            ObjectDesc()
                .id(2)
                .pos({11.0f, 10.0f})
                .type(CellDesc()
                          .frontAngle(180.0f)
                          .cellType(MuscleDesc().mode(AutoCrawlingDesc().maxDistanceDeviation(MaxDistanceDeviation)))
                          .neuralNetwork(nn)),
            ObjectDesc().id(3).pos({12.0f, 10.0f}).type(CellDesc().frontAngle(180.0f)),
        },
        CreatureDesc().id(0),
        GenomeDesc().frontAngle(180.0f));
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);

    auto minDistance = 1.0f;
    auto maxDistance = 1.0f;
    for (int i = 0; i < 500; ++i) {
        _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = actualData.getObjectRef(2);
        auto actualCell1 = actualData.getObjectRef(1);
        auto actualCell3 = actualData.getObjectRef(3);

        ASSERT_EQ(3, actualData._objects.size());

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

TEST_P(MuscleTests_AutoCrawling, muscleWithOneConnection)
{
    auto constexpr MaxDistanceDeviation = 0.5f;

    auto channel0 = GetParam();

    NeuralNetDesc nn;
    nn._weights.clear();
    nn._weights.resize(NEURONS_PER_CELL * NEURONS_PER_CELL, NeuralNetWeight(0));
    nn._biases.at(Channels::CellTypeActivation) = getValue(channel0);

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().frontAngle(0.0f).headCell(true)),
            ObjectDesc()
                .id(2)
                .pos({11.0f, 10.0f})
                .type(CellDesc()
                          .frontAngle(180.0f)
                          .cellType(MuscleDesc().mode(AutoCrawlingDesc().maxDistanceDeviation(MaxDistanceDeviation)))
                          .neuralNetwork(nn)),
        },
        CreatureDesc().id(0),
        GenomeDesc().frontAngle(180.0f));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);

    auto minDistance = 1.0f;
    auto maxDistance = 1.0f;
    for (int i = 0; i < 500; ++i) {
        _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = actualData.getObjectRef(2);
        auto actualCell1 = actualData.getObjectRef(1);

        ASSERT_EQ(2, actualData._objects.size());

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

class MuscleTests_ManualCrawling
    : public MuscleTests
    , public testing::WithParamInterface<Channel0>
{};

INSTANTIATE_TEST_SUITE_P(MuscleTests_ManualCrawling, MuscleTests_ManualCrawling, ::testing::Values(Channel0::Positive, Channel0::Negative, Channel0::Zero));

TEST_P(MuscleTests_ManualCrawling, muscleWithTwoConnections)
{
    auto constexpr MaxDistanceDeviation = 0.5f;

    auto channel0 = GetParam();

    NeuralNetDesc nn;
    nn._weights.clear();
    nn._weights.resize(NEURONS_PER_CELL * NEURONS_PER_CELL, NeuralNetWeight(0));
    nn._biases.at(Channels::CellTypeActivation) = getValue(channel0);

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().frontAngle(0.0f).headCell(true)),
            ObjectDesc()
                .id(2)
                .pos({11.0f, 10.0f})
                .type(CellDesc()
                          .frontAngle(180.0f)
                          .cellType(MuscleDesc().mode(ManualCrawlingDesc().maxDistanceDeviation(MaxDistanceDeviation)))
                          .neuralNetwork(nn)),
            ObjectDesc().id(3).pos({12.0f, 10.0f}).type(CellDesc().frontAngle(180.0f)),
        },
        CreatureDesc().id(0),
        GenomeDesc().frontAngle(180.0f));
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);

    auto minDistance = 1.0f;
    auto maxDistance = 1.0f;
    for (int i = 0; i < 500; ++i) {
        _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = actualData.getObjectRef(2);
        auto actualCell3 = actualData.getObjectRef(3);

        ASSERT_EQ(3, actualData._objects.size());

        auto distance = actualMuscleCell._connections.at(0)._distance;
        EXPECT_TRUE(approxCompare(distance, actualCell3._connections.at(0)._distance));

        minDistance = std::min(minDistance, distance);
        maxDistance = std::max(maxDistance, distance);
    }

    if (channel0 == Channel0::Zero) {
        EXPECT_TRUE(approxCompare(1.0f, minDistance));
        EXPECT_TRUE(approxCompare(1.0f, maxDistance));
    } else if (channel0 == Channel0::Positive) {
        EXPECT_TRUE(approxCompare(1.0f, minDistance));
        EXPECT_TRUE(approxCompare(1.0f + MaxDistanceDeviation, maxDistance));
    } else if (channel0 == Channel0::Negative) {
        EXPECT_TRUE(approxCompare(1.0f - MaxDistanceDeviation, minDistance));
        EXPECT_TRUE(approxCompare(1.0f, maxDistance));
    }
}

TEST_P(MuscleTests_ManualCrawling, muscleWithOneConnection)
{
    auto constexpr MaxDistanceDeviation = 0.5f;

    auto channel0 = GetParam();

    NeuralNetDesc nn;
    nn._weights.clear();
    nn._weights.resize(NEURONS_PER_CELL * NEURONS_PER_CELL, NeuralNetWeight(0));
    nn._biases.at(Channels::CellTypeActivation) = getValue(channel0);

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().frontAngle(0.0f).headCell(true)),
            ObjectDesc()
                .id(2)
                .pos({11.0f, 10.0f})
                .type(CellDesc()
                          .frontAngle(180.0f)
                          .cellType(MuscleDesc().mode(ManualCrawlingDesc().maxDistanceDeviation(MaxDistanceDeviation)))
                          .neuralNetwork(nn)),
        },
        CreatureDesc().id(0),
        GenomeDesc().frontAngle(180.0f));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);

    auto minDistance = 1.0f;
    auto maxDistance = 1.0f;
    for (int i = 0; i < 500; ++i) {
        _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualMuscleCell = actualData.getObjectRef(2);
        auto actualCell1 = actualData.getObjectRef(1);

        ASSERT_EQ(2, actualData._objects.size());

        auto distance = actualMuscleCell._connections.at(0)._distance;
        EXPECT_TRUE(approxCompare(distance, actualCell1._connections.at(0)._distance));

        minDistance = std::min(minDistance, distance);
        maxDistance = std::max(maxDistance, distance);
    }

    if (channel0 == Channel0::Zero) {
        EXPECT_TRUE(approxCompare(1.0f, minDistance));
        EXPECT_TRUE(approxCompare(1.0f, maxDistance));
    } else if (channel0 == Channel0::Positive) {
        EXPECT_TRUE(approxCompare(1.0f, minDistance));
        EXPECT_TRUE(approxCompare(1.0f + MaxDistanceDeviation, maxDistance));
    } else if (channel0 == Channel0::Negative) {
        EXPECT_TRUE(approxCompare(1.0f - MaxDistanceDeviation, minDistance));
        EXPECT_TRUE(approxCompare(1.0f, maxDistance));
    }
}

class MuscleTests_DirectMovement
    : public MuscleTests
    , public testing::WithParamInterface<std::tuple<Channel0, Channel1>>
{};

INSTANTIATE_TEST_SUITE_P(
    MuscleTests_DirectMovement,
    MuscleTests_DirectMovement,
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

TEST_P(MuscleTests_DirectMovement, muscleWithTwoConnections)
{
    auto constexpr AnglePrecision = 3.0f;
    auto [channel0, channel1] = GetParam();

    NeuralNetDesc nn;
    nn._weights.clear();
    nn._weights.resize(NEURONS_PER_CELL * NEURONS_PER_CELL, NeuralNetWeight(0));
    nn._biases.at(Channels::CellTypeActivation) = getValue(channel0);
    nn._biases.at(Channels::MuscleAngle) = getValue(channel1) / 2;

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}),
            ObjectDesc()
                .id(2)
                .pos({11.0f, 10.0f})
                .type(CellDesc().frontAngle(0.0f).headCell(true).cellType(MuscleDesc().mode(DirectMovementDesc())).neuralNetwork(nn)),
            ObjectDesc().id(3).pos({12.0f, 10.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._objects.size());

    auto actualCell1 = actualData.getObjectRef(1);
    auto actualCell2 = actualData.getObjectRef(2);
    auto actualCell3 = actualData.getObjectRef(3);

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
            EXPECT_TRUE(approxCompare(0.0f, Math::getNormalizedAngle(angleVel, -180.0f), AnglePrecision));
        } else if (channel0 == Channel0::Negative) {
            EXPECT_TRUE(approxCompare(180.0f, angleVel, AnglePrecision));
        } else {
            EXPECT_TRUE(approxCompare(0.0f, Math::length(actualCell2._vel), 0.01f));
        }
    } else {
        if (channel0 == Channel0::Positive) {
            EXPECT_TRUE(approxCompare(180.0f, angleVel, AnglePrecision));
        } else if (channel0 == Channel0::Negative) {
            EXPECT_TRUE(approxCompare(0.0f, Math::getNormalizedAngle(angleVel, -180.0f), AnglePrecision));
        } else {
            EXPECT_TRUE(approxCompare(0.0f, Math::length(actualCell2._vel), 0.01f));
        }
    }
}
