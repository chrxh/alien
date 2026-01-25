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
    auto data = Desc().addCreature({
        ObjectDesc().id(1),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getObjectRef(1);
    EXPECT_FALSE(generator.getCellRef()._signalState == SignalState_Active);
}

TEST_F(SignalTests, forwardSignal)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().signal(SignalDesc().channels(signal).numTimesSent(3)).signalState(SignalState_Active)),
        ObjectDesc().id(2).pos({1, 0}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_FALSE(object1.getCellRef()._signalState == SignalState_Active);

    auto object2 = actualData.getObjectRef(2);
    EXPECT_TRUE(object2.getCellRef()._signalState == SignalState_Active);
    EXPECT_EQ(signal, object2.getCellRef()._signal._channels);
    EXPECT_EQ(3, object2.getCellRef()._signal._numTimesSent);
    EXPECT_EQ(1, object1.getCellRef()._signalState);
    EXPECT_EQ(2, object2.getCellRef()._signalState);
}

TEST_F(SignalTests, forwardSignal_detailedPreview)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().signalAndState(signal)),
        ObjectDesc().id(2).pos({1, 0}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setPreviewData(data);
    _simulationFacade->calcTimestepsForPreview(1, true);
    auto actualData = _simulationFacade->getPreviewData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_FALSE(object1.getCellRef()._signalState == SignalState_Active);

    auto object2 = actualData.getObjectRef(2);
    EXPECT_TRUE(object2.getCellRef()._signalState == SignalState_Active);
    EXPECT_EQ(signal, object2.getCellRef()._signal._channels);
    EXPECT_EQ(1, object1.getCellRef()._signalState);
    EXPECT_EQ(2, object2.getCellRef()._signalState);
}

TEST_F(SignalTests, vanishSignal_singleCell)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().signalAndState(signal)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_FALSE(object1.getCellRef()._signalState == SignalState_Active);
}

TEST_F(SignalTests, vanishSignal_relaxationNeeded)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().signal(SignalDesc().channels(signal))),
        ObjectDesc().id(2).pos({1, 0}).type(CellDesc().signalState(1)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_FALSE(object1.getCellRef()._signalState == SignalState_Active);
}

TEST_F(SignalTests, mergeSignals)
{
    std::vector<float> signal1 = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, -1.0f, 0.0f};
    std::vector<float> signal2 = {-0.5f, -1.0f, 0.5f, 1.0f, 0.7f, -0.7f, 0.5f, -0.5f};
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().signal(SignalDesc().channels(signal1).numTimesSent(7)).signalState(SignalState_Active)),
        ObjectDesc().id(2).pos({1, 0}),
        ObjectDesc().id(3).pos({2, 0}).type(CellDesc().signal(SignalDesc().channels(signal2).numTimesSent(3)).signalState(SignalState_Active)),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_FALSE(object1.getCellRef()._signalState == SignalState_Active);

    auto object2 = actualData.getObjectRef(2);
    EXPECT_TRUE(object2.getCellRef()._signalState == SignalState_Active);

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_FALSE(cell3.getCellRef()._signalState == SignalState_Active);

    std::vector<float> sumSignal(signal1.size());
    for (size_t i = 0; i < signal1.size(); ++i) {
        sumSignal[i] = signal1[i] + signal2[i];
    }
    EXPECT_TRUE(approxCompare(sumSignal, object2.getCellRef()._signal._channels));
    EXPECT_EQ(3, object2.getCellRef()._signal._numTimesSent);  // Takes minimum of 7 and 3
    EXPECT_EQ(1, object1.getCellRef()._signalState);
    EXPECT_EQ(2, object2.getCellRef()._signalState);
    EXPECT_EQ(1, cell3.getCellRef()._signalState);
}

TEST_F(SignalTests, forkSignals)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}),
        ObjectDesc().id(2).pos({1, 0}).type(CellDesc().signal(SignalDesc().channels(signal).numTimesSent(5)).signalState(SignalState_Active)),
        ObjectDesc().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_TRUE(object1.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(signal, object1.getCellRef()._signal._channels));
    EXPECT_EQ(5, object1.getCellRef()._signal._numTimesSent);
    EXPECT_EQ(2, object1.getCellRef()._signalState);

    auto object2 = actualData.getObjectRef(2);
    EXPECT_FALSE(object2.getCellRef()._signalState == SignalState_Active);
    EXPECT_EQ(1, object2.getCellRef()._signalState);

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_TRUE(cell3.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(signal, cell3.getCellRef()._signal._channels));
    EXPECT_EQ(5, cell3.getCellRef()._signal._numTimesSent);
    EXPECT_EQ(2, cell3.getCellRef()._signalState);
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
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}),
        ObjectDesc().id(2).pos({1, 0}).type(CellDesc().signalAndState(signal).signalRestriction(side == AngleRange::Start ? -44.0f : 44.0f, 90.0f)),
        ObjectDesc().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_FALSE(object1.getCellRef()._signalState == SignalState_Active);
    EXPECT_EQ(0, object1.getCellRef()._signalState);

    auto object2 = actualData.getObjectRef(2);
    EXPECT_FALSE(object2.getCellRef()._signalState == SignalState_Active);
    EXPECT_EQ(1, object2.getCellRef()._signalState);

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_TRUE(cell3.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(signal, cell3.getCellRef()._signal._channels));
    EXPECT_EQ(2, cell3.getCellRef()._signalState);
}

TEST_P(SignalTests_BothSides, routeSignalOnRight_sharpMismatch)
{
    auto side = GetParam();
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}),
        ObjectDesc().id(2).pos({1, 0}).type(CellDesc().signalAndState(signal).signalRestriction(side == AngleRange::Start ? -45.0f : 45.0f, 90.0f)),
        ObjectDesc().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_FALSE(object1.getCellRef()._signalState == SignalState_Active);

    auto object2 = actualData.getObjectRef(2);
    EXPECT_FALSE(object2.getCellRef()._signalState == SignalState_Active);

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_FALSE(cell3.getCellRef()._signalState == SignalState_Active);
}

// Tests for SignalRestrictionMode_Conditional

TEST_F(SignalTests, conditionalMode_outsideCone_alwaysBlocked)
{
    // Conditional mode: signals outside the cone are always blocked, regardless of channel[0]
    std::vector<float> signal = {0.5f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};  // channel[0] = 0.5 >= 0

    // Cell 2 has conditional restriction - signal to cell 3 is outside the cone
    auto cell2Desc = ObjectDesc().id(2).pos({1, 0}).type(CellDesc().signalAndState(signal));
    cell2Desc.getCellRef()._signalRestriction._mode = SignalRestrictionMode_Conditional;
    cell2Desc.getCellRef()._signalRestriction._baseAngle = 45.0f;  // Restriction points away from cell 3
    cell2Desc.getCellRef()._signalRestriction._openingAngle = 90.0f;

    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}),
        cell2Desc,
        ObjectDesc().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_FALSE(cell3.getCellRef()._signalState == SignalState_Active);  // Signal blocked because outside cone
}

TEST_F(SignalTests, conditionalMode_insideCone_channel0Negative_blocked)
{
    // Conditional mode: signals inside the cone are blocked if channel[0] < 0
    std::vector<float> signal = {-0.5f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};  // channel[0] = -0.5 < 0

    // Cell 2 has conditional restriction - signal to cell 3 is inside the cone but channel[0] < 0
    auto cell2Desc = ObjectDesc().id(2).pos({1, 0}).type(CellDesc().signalAndState(signal));
    cell2Desc.getCellRef()._signalRestriction._mode = SignalRestrictionMode_Conditional;
    cell2Desc.getCellRef()._signalRestriction._baseAngle = 0.0f;  // Restriction centered on cell 3 direction
    cell2Desc.getCellRef()._signalRestriction._openingAngle = 90.0f;

    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}),
        cell2Desc,
        ObjectDesc().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_FALSE(cell3.getCellRef()._signalState == SignalState_Active);  // Signal blocked because channel[0] < 0
}

TEST_F(SignalTests, conditionalMode_insideCone_channel0Zero_passes)
{
    // Conditional mode: signals inside the cone pass if channel[0] >= 0
    std::vector<float> signal = {0.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};  // channel[0] = 0 >= 0

    // Cell 2 has conditional restriction - signal to cell 3 is inside the cone and channel[0] >= 0
    auto cell2Desc = ObjectDesc().id(2).pos({1, 0}).type(CellDesc().signalAndState(signal));
    cell2Desc.getCellRef()._signalRestriction._mode = SignalRestrictionMode_Conditional;
    cell2Desc.getCellRef()._signalRestriction._baseAngle = 0.0f;  // Restriction centered on cell 3 direction
    cell2Desc.getCellRef()._signalRestriction._openingAngle = 90.0f;

    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}),
        cell2Desc,
        ObjectDesc().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_TRUE(cell3.getCellRef()._signalState == SignalState_Active);  // Signal passes through
    EXPECT_TRUE(approxCompare(signal, cell3.getCellRef()._signal._channels));
}

TEST_F(SignalTests, conditionalMode_insideCone_channel0Positive_passes)
{
    // Conditional mode: signals inside the cone pass if channel[0] >= 0
    std::vector<float> signal = {0.5f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};  // channel[0] = 0.5 >= 0

    // Cell 2 has conditional restriction - signal to cell 3 is inside the cone and channel[0] >= 0
    auto cell2Desc = ObjectDesc().id(2).pos({1, 0}).type(CellDesc().signalAndState(signal));
    cell2Desc.getCellRef()._signalRestriction._mode = SignalRestrictionMode_Conditional;
    cell2Desc.getCellRef()._signalRestriction._baseAngle = 0.0f;  // Restriction centered on cell 3 direction
    cell2Desc.getCellRef()._signalRestriction._openingAngle = 90.0f;

    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}),
        cell2Desc,
        ObjectDesc().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_TRUE(cell3.getCellRef()._signalState == SignalState_Active);  // Signal passes through
    EXPECT_TRUE(approxCompare(signal, cell3.getCellRef()._signal._channels));
}

TEST_F(SignalTests, inactiveMode_noRestriction)
{
    // Inactive mode means no restriction regardless of angles
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};

    // Cell 2 has inactive restriction - signal should pass even with restrictive angles
    auto cell2Desc = ObjectDesc().id(2).pos({1, 0}).type(CellDesc().signalAndState(signal));
    cell2Desc.getCellRef()._signalRestriction._mode = SignalRestrictionMode_Inactive;
    cell2Desc.getCellRef()._signalRestriction._baseAngle = 45.0f;  // Would block if active
    cell2Desc.getCellRef()._signalRestriction._openingAngle = 10.0f;

    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}),
        cell2Desc,
        ObjectDesc().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_TRUE(cell3.getCellRef()._signalState == SignalState_Active);  // Signal passes through
    EXPECT_TRUE(approxCompare(signal, cell3.getCellRef()._signal._channels));
}

// Tests for edge cases with baseAngle normalization
// These test cases verify that signal propagation works correctly when baseAngle
// causes the restriction cone to wrap around angle boundaries (e.g., negative angles
// or angles > 360)

TEST_F(SignalTests, signalRestriction_largeBaseAngle_signalPasses_toFirstConnection)
{
    // baseAngle = 170 with openingAngle = 90 should allow signal to cell 1
    // Unnormalized cone range: 180 + 170 - 45 = 305 to 180 + 170 + 45 = 395
    // Normalized cone range: 305 to 35 (wrapping around 0)
    // Connection to cell 1 is at connectionAngle 0 (first connection), which is inside the cone
    //
    // NOTE: Without normalization, isAngleInBetween(305, 395, 0) incorrectly returns false
    // because the algorithm doesn't handle angle2 > 360 properly for the candidate angle check.
    // With normalization to (305, 35), isAngleInBetween correctly returns true.
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};

    auto cell2Desc = ObjectDesc().id(2).pos({1, 0}).type(CellDesc().signalAndState(signal));
    cell2Desc.getCellRef()._signalRestriction._mode = SignalRestrictionMode_Active;
    cell2Desc.getCellRef()._signalRestriction._baseAngle = 170.0f;
    cell2Desc.getCellRef()._signalRestriction._openingAngle = 90.0f;

    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}),
        cell2Desc,
        ObjectDesc().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell1 = actualData.getObjectRef(1);
    EXPECT_TRUE(cell1.getCellRef()._signalState == SignalState_Active);  // Signal should pass to cell 1
    EXPECT_TRUE(approxCompare(signal, cell1.getCellRef()._signal._channels));

    // Cell 3 should NOT receive signal (connectionAngle 180 is outside cone 305-35)
    auto cell3 = actualData.getObjectRef(3);
    EXPECT_FALSE(cell3.getCellRef()._signalState == SignalState_Active);
}

TEST_F(SignalTests, signalRestriction_largeNegativeBaseAngle_signalPasses_toFirstConnection)
{
    // baseAngle = -170 with openingAngle = 90 should allow signal to cell 1
    // Unnormalized cone range: 180 + (-170) - 45 = -35 to 180 + (-170) + 45 = 55
    // Normalized cone range: 325 to 55 (wrapping around 0)
    // Connection to cell 1 is at connectionAngle 0 (first connection), which is inside the cone
    //
    // NOTE: Without normalization, isAngleInBetween(-35, 55, 0) may give incorrect results
    // because the algorithm assumes angle1 is in range [0, 360).
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};

    auto cell2Desc = ObjectDesc().id(2).pos({1, 0}).type(CellDesc().signalAndState(signal));
    cell2Desc.getCellRef()._signalRestriction._mode = SignalRestrictionMode_Active;
    cell2Desc.getCellRef()._signalRestriction._baseAngle = -170.0f;
    cell2Desc.getCellRef()._signalRestriction._openingAngle = 90.0f;

    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}),
        cell2Desc,
        ObjectDesc().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell1 = actualData.getObjectRef(1);
    EXPECT_TRUE(cell1.getCellRef()._signalState == SignalState_Active);  // Signal should pass to cell 1
    EXPECT_TRUE(approxCompare(signal, cell1.getCellRef()._signal._channels));

    // Cell 3 should NOT receive signal (connectionAngle 180 is outside cone 325-55)
    auto cell3 = actualData.getObjectRef(3);
    EXPECT_FALSE(cell3.getCellRef()._signalState == SignalState_Active);
}

TEST_F(SignalTests, signalRestriction_largeBaseAngle_signalBlocked)
{
    // baseAngle = 90 with openingAngle = 90 should block signal to cell 1
    // Cone range: 180 + 90 - 45 = 225 to 180 + 90 + 45 = 315
    // Connection to cell 1 is at connectionAngle 0 (first connection), which is outside the cone
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};

    auto cell2Desc = ObjectDesc().id(2).pos({1, 0}).type(CellDesc().signalAndState(signal));
    cell2Desc.getCellRef()._signalRestriction._mode = SignalRestrictionMode_Active;
    cell2Desc.getCellRef()._signalRestriction._baseAngle = 90.0f;
    cell2Desc.getCellRef()._signalRestriction._openingAngle = 90.0f;

    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}),
        cell2Desc,
        ObjectDesc().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell1 = actualData.getObjectRef(1);
    EXPECT_FALSE(cell1.getCellRef()._signalState == SignalState_Active);  // Signal should be blocked

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_FALSE(cell3.getCellRef()._signalState == SignalState_Active);  // Also blocked
}