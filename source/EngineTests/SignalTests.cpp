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