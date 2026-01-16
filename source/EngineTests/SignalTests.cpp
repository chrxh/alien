#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class SignalTests : public IntegrationTestFramework
{
public:
    SignalTests()
        : IntegrationTestFramework()
    {}

    ~SignalTests() = default;
};

TEST_F(SignalTests, noSignal)
{
    Description data;
    data._objects = {
        ObjectDescription().id(1),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getObjectRef(1);
    EXPECT_FALSE(std::get<CellDescription>(generator._type)._signalState == SignalState_Active);
}

TEST_F(SignalTests, forwardSignal)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({0, 0}).type(CellDescription().signal(SignalDescription().channels(signal).numTimesSent(3)).signalState(SignalState_Active)),
        ObjectDescription().id(2).pos({1, 0}),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_FALSE(std::get<CellDescription>(object1._type)._signalState == SignalState_Active);

    auto object2 = actualData.getObjectRef(2);
    EXPECT_TRUE(std::get<CellDescription>(object2._type)._signalState == SignalState_Active);
    EXPECT_EQ(signal, std::get<CellDescription>(object2._type)._signal._channels);
    EXPECT_EQ(3, std::get<CellDescription>(object2._type)._signal._numTimesSent);
    EXPECT_EQ(1, std::get<CellDescription>(object1._type)._signalState);
    EXPECT_EQ(2, std::get<CellDescription>(object2._type)._signalState);
}

TEST_F(SignalTests, forwardSignal_detailedPreview)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({0, 0}).type(CellDescription().signalAndState(signal)),
        ObjectDescription().id(2).pos({1, 0}),
    };
    data.addConnection(1, 2);

    _simulationFacade->setPreviewData(data);
    _simulationFacade->calcTimestepsForPreview(1, true);
    auto actualData = _simulationFacade->getPreviewData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_FALSE(std::get<CellDescription>(object1._type)._signalState == SignalState_Active);

    auto object2 = actualData.getObjectRef(2);
    EXPECT_TRUE(std::get<CellDescription>(object2._type)._signalState == SignalState_Active);
    EXPECT_EQ(signal, std::get<CellDescription>(object2._type)._signal._channels);
    EXPECT_EQ(1, std::get<CellDescription>(object1._type)._signalState);
    EXPECT_EQ(2, std::get<CellDescription>(object2._type)._signalState);
}

TEST_F(SignalTests, vanishSignal_singleCell)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({0, 0}).type(CellDescription().signalAndState(signal)),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_FALSE(std::get<CellDescription>(object1._type)._signalState == SignalState_Active);
}

TEST_F(SignalTests, vanishSignal_relaxationNeeded)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({0, 0}).type(CellDescription().signal(SignalDescription().channels(signal))),
        ObjectDescription().id(2).pos({1, 0}).type(CellDescription().signalState(1)),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_FALSE(std::get<CellDescription>(object1._type)._signalState == SignalState_Active);
}

TEST_F(SignalTests, mergeSignals)
{
    std::vector<float> signal1 = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, -1.0f, 0.0f};
    std::vector<float> signal2 = {-0.5f, -1.0f, 0.5f, 1.0f, 0.7f, -0.7f, 0.5f, -0.5f};
    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({0, 0}).type(CellDescription().signal(SignalDescription().channels(signal1).numTimesSent(7)).signalState(SignalState_Active)),
        ObjectDescription().id(2).pos({1, 0}),
        ObjectDescription().id(3).pos({2, 0}).type(CellDescription().signal(SignalDescription().channels(signal2).numTimesSent(3)).signalState(SignalState_Active)),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_FALSE(std::get<CellDescription>(object1._type)._signalState == SignalState_Active);

    auto object2 = actualData.getObjectRef(2);
    EXPECT_TRUE(std::get<CellDescription>(object2._type)._signalState == SignalState_Active);

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_FALSE(std::get<CellDescription>(cell3._type)._signalState == SignalState_Active);

    std::vector<float> sumSignal(signal1.size());
    for (size_t i = 0; i < signal1.size(); ++i) {
        sumSignal[i] = signal1[i] + signal2[i];
    }
    EXPECT_TRUE(approxCompare(sumSignal, std::get<CellDescription>(object2._type)._signal._channels));
    EXPECT_EQ(3, std::get<CellDescription>(object2._type)._signal._numTimesSent);  // Takes minimum of 7 and 3
    EXPECT_EQ(1, std::get<CellDescription>(object1._type)._signalState);
    EXPECT_EQ(2, std::get<CellDescription>(object2._type)._signalState);
    EXPECT_EQ(1, std::get<CellDescription>(cell3._type)._signalState);
}

TEST_F(SignalTests, forkSignals)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};
    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({0, 0}),
        ObjectDescription().id(2).pos({1, 0}).type(CellDescription().signal(SignalDescription().channels(signal).numTimesSent(5)).signalState(SignalState_Active)),
        ObjectDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_TRUE(std::get<CellDescription>(object1._type)._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(signal, std::get<CellDescription>(object1._type)._signal._channels));
    EXPECT_EQ(5, std::get<CellDescription>(object1._type)._signal._numTimesSent);
    EXPECT_EQ(2, std::get<CellDescription>(object1._type)._signalState);

    auto object2 = actualData.getObjectRef(2);
    EXPECT_FALSE(std::get<CellDescription>(object2._type)._signalState == SignalState_Active);
    EXPECT_EQ(1, std::get<CellDescription>(object2._type)._signalState);

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_TRUE(std::get<CellDescription>(cell3._type)._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(signal, std::get<CellDescription>(cell3._type)._signal._channels));
    EXPECT_EQ(5, std::get<CellDescription>(cell3._type)._signal._numTimesSent);
    EXPECT_EQ(2, std::get<CellDescription>(cell3._type)._signalState);
}

enum class AngleRange
{
    Start,
    End
};

class SignalTests_BothSides
    : public SignalTests
    , public testing::WithParamInterface<AngleRange>
{};

INSTANTIATE_TEST_SUITE_P(SignalTests_BothSides, SignalTests_BothSides, ::testing::Values(AngleRange::Start, AngleRange::End));

TEST_P(SignalTests_BothSides, routeSignalOnRight_sharpMatch)
{
    auto side = GetParam();
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};
    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({0, 0}),
        ObjectDescription().id(2).pos({1, 0}).type(CellDescription().signalAndState(signal).signalRestriction(side == AngleRange::Start ? -44.0f : 44.0f, 90.0f)),
        ObjectDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_FALSE(std::get<CellDescription>(object1._type)._signalState == SignalState_Active);
    EXPECT_EQ(0, std::get<CellDescription>(object1._type)._signalState);

    auto object2 = actualData.getObjectRef(2);
    EXPECT_FALSE(std::get<CellDescription>(object2._type)._signalState == SignalState_Active);
    EXPECT_EQ(1, std::get<CellDescription>(object2._type)._signalState);

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_TRUE(std::get<CellDescription>(cell3._type)._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(signal, std::get<CellDescription>(cell3._type)._signal._channels));
    EXPECT_EQ(2, std::get<CellDescription>(cell3._type)._signalState);
}

TEST_P(SignalTests_BothSides, routeSignalOnRight_sharpMismatch)
{
    auto side = GetParam();
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};
    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({0, 0}),
        ObjectDescription().id(2).pos({1, 0}).type(CellDescription().signalAndState(signal).signalRestriction(side == AngleRange::Start ? -45.0f : 45.0f, 90.0f)),
        ObjectDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_FALSE(std::get<CellDescription>(object1._type)._signalState == SignalState_Active);

    auto object2 = actualData.getObjectRef(2);
    EXPECT_FALSE(std::get<CellDescription>(object2._type)._signalState == SignalState_Active);

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_FALSE(std::get<CellDescription>(cell3._type)._signalState == SignalState_Active);
}

// Tests for SignalRestrictionMode_Conditional

TEST_F(SignalTests, conditionalMode_outsideCone_alwaysBlocked)
{
    // Conditional mode: signals outside the cone are always blocked, regardless of channel[0]
    std::vector<float> signal = {0.5f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};  // channel[0] = 0.5 >= 0
    Description data;

    // Cell 2 has conditional restriction - signal to cell 3 is outside the cone
    auto cell2Desc = ObjectDescription().id(2).pos({1, 0}).type(CellDescription().signalAndState(signal));
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._mode = SignalRestrictionMode_Conditional;
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._baseAngle = 45.0f;  // Restriction points away from cell 3
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._openingAngle = 90.0f;

    data._objects = {
        ObjectDescription().id(1).pos({0, 0}),
        cell2Desc,
        ObjectDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_FALSE(std::get<CellDescription>(cell3._type)._signalState == SignalState_Active);  // Signal blocked because outside cone
}

TEST_F(SignalTests, conditionalMode_insideCone_channel0Negative_blocked)
{
    // Conditional mode: signals inside the cone are blocked if channel[0] < 0
    std::vector<float> signal = {-0.5f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};  // channel[0] = -0.5 < 0
    Description data;

    // Cell 2 has conditional restriction - signal to cell 3 is inside the cone but channel[0] < 0
    auto cell2Desc = ObjectDescription().id(2).pos({1, 0}).type(CellDescription().signalAndState(signal));
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._mode = SignalRestrictionMode_Conditional;
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._baseAngle = 0.0f;  // Restriction centered on cell 3 direction
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._openingAngle = 90.0f;

    data._objects = {
        ObjectDescription().id(1).pos({0, 0}),
        cell2Desc,
        ObjectDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_FALSE(std::get<CellDescription>(cell3._type)._signalState == SignalState_Active);  // Signal blocked because channel[0] < 0
}

TEST_F(SignalTests, conditionalMode_insideCone_channel0Zero_passes)
{
    // Conditional mode: signals inside the cone pass if channel[0] >= 0
    std::vector<float> signal = {0.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};  // channel[0] = 0 >= 0
    Description data;

    // Cell 2 has conditional restriction - signal to cell 3 is inside the cone and channel[0] >= 0
    auto cell2Desc = ObjectDescription().id(2).pos({1, 0}).type(CellDescription().signalAndState(signal));
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._mode = SignalRestrictionMode_Conditional;
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._baseAngle = 0.0f;  // Restriction centered on cell 3 direction
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._openingAngle = 90.0f;

    data._objects = {
        ObjectDescription().id(1).pos({0, 0}),
        cell2Desc,
        ObjectDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_TRUE(std::get<CellDescription>(cell3._type)._signalState == SignalState_Active);  // Signal passes through
    EXPECT_TRUE(approxCompare(signal, std::get<CellDescription>(cell3._type)._signal._channels));
}

TEST_F(SignalTests, conditionalMode_insideCone_channel0Positive_passes)
{
    // Conditional mode: signals inside the cone pass if channel[0] >= 0
    std::vector<float> signal = {0.5f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};  // channel[0] = 0.5 >= 0
    Description data;

    // Cell 2 has conditional restriction - signal to cell 3 is inside the cone and channel[0] >= 0
    auto cell2Desc = ObjectDescription().id(2).pos({1, 0}).type(CellDescription().signalAndState(signal));
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._mode = SignalRestrictionMode_Conditional;
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._baseAngle = 0.0f;  // Restriction centered on cell 3 direction
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._openingAngle = 90.0f;

    data._objects = {
        ObjectDescription().id(1).pos({0, 0}),
        cell2Desc,
        ObjectDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_TRUE(std::get<CellDescription>(cell3._type)._signalState == SignalState_Active);  // Signal passes through
    EXPECT_TRUE(approxCompare(signal, std::get<CellDescription>(cell3._type)._signal._channels));
}

TEST_F(SignalTests, inactiveMode_noRestriction)
{
    // Inactive mode means no restriction regardless of angles
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};
    Description data;

    // Cell 2 has inactive restriction - signal should pass even with restrictive angles
    auto cell2Desc = ObjectDescription().id(2).pos({1, 0}).type(CellDescription().signalAndState(signal));
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._mode = SignalRestrictionMode_Inactive;
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._baseAngle = 45.0f;  // Would block if active
    std::get<CellDescription>(cell2Desc._type)._signalRestriction._openingAngle = 10.0f;

    data._objects = {
        ObjectDescription().id(1).pos({0, 0}),
        cell2Desc,
        ObjectDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_TRUE(std::get<CellDescription>(cell3._type)._signalState == SignalState_Active);  // Signal passes through
    EXPECT_TRUE(approxCompare(signal, std::get<CellDescription>(cell3._type)._signal._channels));
}