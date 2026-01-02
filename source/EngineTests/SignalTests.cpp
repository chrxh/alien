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
    data._cells = {
        CellDescription().id(1),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getCellRef(1);
    EXPECT_FALSE(generator._signalState == SignalState_Active);
}

TEST_F(SignalTests, forwardSignal)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    Description data;
    data._cells = {
        CellDescription().id(1).pos({0, 0}).signalAndState(signal),
        CellDescription().id(2).pos({1, 0}),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell1 = actualData.getCellRef(1);
    EXPECT_FALSE(cell1._signalState == SignalState_Active);

    auto cell2 = actualData.getCellRef(2);
    EXPECT_TRUE(cell2._signalState == SignalState_Active);
    EXPECT_EQ(signal, cell2._signal._channels);
    EXPECT_EQ(1, cell1._signalState);
    EXPECT_EQ(2, cell2._signalState);
}

TEST_F(SignalTests, forwardSignal_detailedPreview)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    Description data;
    data._cells = {
        CellDescription().id(1).pos({0, 0}).signalAndState(signal),
        CellDescription().id(2).pos({1, 0}),
    };
    data.addConnection(1, 2);

    _simulationFacade->setPreviewData(data);
    _simulationFacade->calcTimestepsForPreview(1, true);
    auto actualData = _simulationFacade->getPreviewData();

    auto cell1 = actualData.getCellRef(1);
    EXPECT_FALSE(cell1._signalState == SignalState_Active);

    auto cell2 = actualData.getCellRef(2);
    EXPECT_TRUE(cell2._signalState == SignalState_Active);
    EXPECT_EQ(signal, cell2._signal._channels);
    EXPECT_EQ(1, cell1._signalState);
    EXPECT_EQ(2, cell2._signalState);
}

TEST_F(SignalTests, vanishSignal_singleCell)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    Description data;
    data._cells = {
        CellDescription().id(1).pos({0, 0}).signalAndState(signal),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell1 = actualData.getCellRef(1);
    EXPECT_FALSE(cell1._signalState == SignalState_Active);
}

TEST_F(SignalTests, vanishSignal_relaxationNeeded)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    Description data;
    data._cells = {
        CellDescription().id(1).pos({0, 0}).signal(SignalDescription().channels(signal)),
        CellDescription().id(2).pos({1, 0}).signalState(1),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell1 = actualData.getCellRef(1);
    EXPECT_FALSE(cell1._signalState == SignalState_Active);
}

TEST_F(SignalTests, mergeSignals)
{
    std::vector<float> signal1 = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, -1.0f, 0.0f};
    std::vector<float> signal2 = {-0.5f, -1.0f, 0.5f, 1.0f, 0.7f, -0.7f, 0.5f, -0.5f};
    Description data;
    data._cells = {
        CellDescription().id(1).pos({0, 0}).signalAndState(signal1),
        CellDescription().id(2).pos({1, 0}),
        CellDescription().id(3).pos({2, 0}).signalAndState(signal2),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell1 = actualData.getCellRef(1);
    EXPECT_FALSE(cell1._signalState == SignalState_Active);

    auto cell2 = actualData.getCellRef(2);
    EXPECT_TRUE(cell2._signalState == SignalState_Active);

    auto cell3 = actualData.getCellRef(3);
    EXPECT_FALSE(cell3._signalState == SignalState_Active);

    std::vector<float> sumSignal(signal1.size());
    for (size_t i = 0; i < signal1.size(); ++i) {
        sumSignal[i] = signal1[i] + signal2[i];
    }
    EXPECT_TRUE(approxCompare(sumSignal, cell2._signal._channels));
    EXPECT_EQ(1, cell1._signalState);
    EXPECT_EQ(2, cell2._signalState);
    EXPECT_EQ(1, cell3._signalState);
}

TEST_F(SignalTests, forkSignals)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};
    Description data;
    data._cells = {
        CellDescription().id(1).pos({0, 0}),
        CellDescription().id(2).pos({1, 0}).signalAndState(signal),
        CellDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell1 = actualData.getCellRef(1);
    EXPECT_TRUE(cell1._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(signal, cell1._signal._channels));
    EXPECT_EQ(2, cell1._signalState);

    auto cell2 = actualData.getCellRef(2);
    EXPECT_FALSE(cell2._signalState == SignalState_Active);
    EXPECT_EQ(1, cell2._signalState);

    auto cell3 = actualData.getCellRef(3);
    EXPECT_TRUE(cell3._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(signal, cell3._signal._channels));
    EXPECT_EQ(2, cell3._signalState);
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
    data._cells = {
        CellDescription().id(1).pos({0, 0}),
        CellDescription().id(2).pos({1, 0}).signalAndState(signal).signalRestriction(side == AngleRange::Start ? -44.0f : 44.0f, 90.0f),
        CellDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell1 = actualData.getCellRef(1);
    EXPECT_FALSE(cell1._signalState == SignalState_Active);
    EXPECT_EQ(0, cell1._signalState);

    auto cell2 = actualData.getCellRef(2);
    EXPECT_FALSE(cell2._signalState == SignalState_Active);
    EXPECT_EQ(1, cell2._signalState);

    auto cell3 = actualData.getCellRef(3);
    EXPECT_TRUE(cell3._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(signal, cell3._signal._channels));
    EXPECT_EQ(2, cell3._signalState);
}

TEST_P(SignalTests_BothSides, routeSignalOnRight_sharpMismatch)
{
    auto side = GetParam();
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};
    Description data;
    data._cells = {
        CellDescription().id(1).pos({0, 0}),
        CellDescription().id(2).pos({1, 0}).signalAndState(signal).signalRestriction(side == AngleRange::Start ? -45.0f : 45.0f, 90.0f),
        CellDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell1 = actualData.getCellRef(1);
    EXPECT_FALSE(cell1._signalState == SignalState_Active);

    auto cell2 = actualData.getCellRef(2);
    EXPECT_FALSE(cell2._signalState == SignalState_Active);

    auto cell3 = actualData.getCellRef(3);
    EXPECT_FALSE(cell3._signalState == SignalState_Active);
}

// Tests for SignalRestrictionMode_Conditional

TEST_F(SignalTests, conditionalMode_channel0Positive_restrictionApplied)
{
    // Conditional mode with channel[0] >= 0 means restriction should be applied
    std::vector<float> signal = {0.5f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};  // channel[0] = 0.5 >= 0
    Description data;

    // Cell 2 has conditional restriction - signal should be blocked because angle is outside restriction
    auto cell2Desc = CellDescription().id(2).pos({1, 0}).signalAndState(signal);
    cell2Desc._signalRestriction._mode = SignalRestrictionMode_Conditional;
    cell2Desc._signalRestriction._baseAngle = 45.0f;  // Restriction points away from cell 3
    cell2Desc._signalRestriction._openingAngle = 90.0f;

    data._cells = {
        CellDescription().id(1).pos({0, 0}),
        cell2Desc,
        CellDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getCellRef(3);
    EXPECT_FALSE(cell3._signalState == SignalState_Active);  // Signal blocked by restriction
}

TEST_F(SignalTests, conditionalMode_channel0Negative_noRestriction)
{
    // Conditional mode with channel[0] < 0 means no restriction should be applied
    std::vector<float> signal = {-0.5f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};  // channel[0] = -0.5 < 0
    Description data;

    // Cell 2 has conditional restriction - but signal should pass because channel[0] < 0
    auto cell2Desc = CellDescription().id(2).pos({1, 0}).signalAndState(signal);
    cell2Desc._signalRestriction._mode = SignalRestrictionMode_Conditional;
    cell2Desc._signalRestriction._baseAngle = 45.0f;  // Restriction points away from cell 3
    cell2Desc._signalRestriction._openingAngle = 90.0f;

    data._cells = {
        CellDescription().id(1).pos({0, 0}),
        cell2Desc,
        CellDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getCellRef(3);
    EXPECT_TRUE(cell3._signalState == SignalState_Active);  // Signal passes through
    EXPECT_TRUE(approxCompare(signal, cell3._signal._channels));
}

TEST_F(SignalTests, conditionalMode_channel0Zero_restrictionApplied)
{
    // Conditional mode with channel[0] = 0 means restriction should be applied (>= 0)
    std::vector<float> signal = {0.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};  // channel[0] = 0 >= 0
    Description data;

    // Cell 2 has conditional restriction - restriction should be applied
    auto cell2Desc = CellDescription().id(2).pos({1, 0}).signalAndState(signal);
    cell2Desc._signalRestriction._mode = SignalRestrictionMode_Conditional;
    cell2Desc._signalRestriction._baseAngle = 45.0f;  // Restriction points away from cell 3
    cell2Desc._signalRestriction._openingAngle = 90.0f;

    data._cells = {
        CellDescription().id(1).pos({0, 0}),
        cell2Desc,
        CellDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getCellRef(3);
    EXPECT_FALSE(cell3._signalState == SignalState_Active);  // Signal blocked by restriction
}

TEST_F(SignalTests, conditionalMode_passesWithinCone)
{
    // Conditional mode with channel[0] >= 0 and angle within cone - signal should pass
    std::vector<float> signal = {0.5f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};  // channel[0] = 0.5 >= 0
    Description data;

    // Cell 2 has conditional restriction - signal should pass because angle is within restriction cone
    auto cell2Desc = CellDescription().id(2).pos({1, 0}).signalAndState(signal);
    cell2Desc._signalRestriction._mode = SignalRestrictionMode_Conditional;
    cell2Desc._signalRestriction._baseAngle = 0.0f;  // Restriction centered on cell 3 direction
    cell2Desc._signalRestriction._openingAngle = 90.0f;

    data._cells = {
        CellDescription().id(1).pos({0, 0}),
        cell2Desc,
        CellDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getCellRef(3);
    EXPECT_TRUE(cell3._signalState == SignalState_Active);  // Signal passes through
    EXPECT_TRUE(approxCompare(signal, cell3._signal._channels));
}

TEST_F(SignalTests, inactiveMode_noRestriction)
{
    // Inactive mode means no restriction regardless of angles
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};
    Description data;

    // Cell 2 has inactive restriction - signal should pass even with restrictive angles
    auto cell2Desc = CellDescription().id(2).pos({1, 0}).signalAndState(signal);
    cell2Desc._signalRestriction._mode = SignalRestrictionMode_Inactive;
    cell2Desc._signalRestriction._baseAngle = 45.0f;  // Would block if active
    cell2Desc._signalRestriction._openingAngle = 10.0f;

    data._cells = {
        CellDescription().id(1).pos({0, 0}),
        cell2Desc,
        CellDescription().id(3).pos({2, 0}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getCellRef(3);
    EXPECT_TRUE(cell3._signalState == SignalState_Active);  // Signal passes through
    EXPECT_TRUE(approxCompare(signal, cell3._signal._channels));
}