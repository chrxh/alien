#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class SignalTests_New : public IntegrationTestFramework
{
public:
    SignalTests_New()
        : IntegrationTestFramework()
    {}

    ~SignalTests_New() = default;
};

TEST_F(SignalTests_New, noSignal)
{
    auto data = DataDescription().addCells({
        CellDescription().id(1),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    EXPECT_FALSE(oscillator._signal.has_value());
}

TEST_F(SignalTests_New, forwardSignal)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    auto data = DataDescription().addCells({
        CellDescription().id(1).pos({0, 0}).signal(signal),
        CellDescription().id(2).pos({1, 0}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto cell1 = actualCellById.at(1);
    EXPECT_FALSE(cell1._signal.has_value());

    auto cell2 = actualCellById.at(2);
    EXPECT_TRUE(cell2._signal.has_value());
    EXPECT_EQ(signal, cell2._signal->_channels);
    EXPECT_EQ(1, cell1._signalRelaxationTime);
    EXPECT_EQ(2, cell2._signalRelaxationTime);
}

TEST_F(SignalTests_New, vanishSignal_singleCell)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    auto data = DataDescription().addCells({
        CellDescription().id(1).pos({0, 0}).signal(signal),
    });
    
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto cell1 = actualCellById.at(1);
    EXPECT_FALSE(cell1._signal.has_value());
}

TEST_F(SignalTests_New, vanishSignal_relaxationNeeded)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    auto data = DataDescription().addCells({
        CellDescription().id(1).pos({0, 0}).signal(SignalDescription().channels(signal)),
        CellDescription().id(2).pos({1, 0}).signalRelaxationTime(1),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto cell1 = actualCellById.at(1);
    EXPECT_FALSE(cell1._signal.has_value());
}

TEST_F(SignalTests_New, mergeSignals)
{
    std::vector<float> signal1 = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, -1.0f, 0.0f};
    std::vector<float> signal2 = {-0.5f, -1.0f, 0.5f, 1.0f, 0.7f, -0.7f, 0.5f, -0.5f};
    auto data = DataDescription().addCells({
        CellDescription().id(1).pos({0, 0}).signal(signal1),
        CellDescription().id(2).pos({1, 0}),
        CellDescription().id(3).pos({2, 0}).signal(signal2),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto cell1 = actualCellById.at(1);
    EXPECT_FALSE(cell1._signal.has_value());

    auto cell2 = actualCellById.at(2);
    EXPECT_TRUE(cell2._signal.has_value());

    auto cell3 = actualCellById.at(3);
    EXPECT_FALSE(cell3._signal.has_value());

    std::vector<float> sumSignal(signal1.size());
    for (size_t i = 0; i < signal1.size(); ++i) {
        sumSignal[i] = signal1[i] + signal2[i];
    }
    EXPECT_TRUE(approxCompare(sumSignal, cell2._signal->_channels));
    EXPECT_EQ(1, cell1._signalRelaxationTime);
    EXPECT_EQ(2, cell2._signalRelaxationTime);
    EXPECT_EQ(1, cell3._signalRelaxationTime);
}

TEST_F(SignalTests_New, forkSignals)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};
    auto data = DataDescription().addCells({
        CellDescription().id(1).pos({0, 0}),
        CellDescription().id(2).pos({1, 0}).signal(signal),
        CellDescription().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto cell1 = actualCellById.at(1);
    EXPECT_TRUE(cell1._signal.has_value());
    EXPECT_TRUE(approxCompare(signal, cell1._signal->_channels));
    EXPECT_EQ(2, cell1._signalRelaxationTime);

    auto cell2 = actualCellById.at(2);
    EXPECT_FALSE(cell2._signal.has_value());
    EXPECT_EQ(1, cell2._signalRelaxationTime);

    auto cell3 = actualCellById.at(3);
    EXPECT_TRUE(cell3._signal.has_value());
    EXPECT_TRUE(approxCompare(signal, cell3._signal->_channels));
    EXPECT_EQ(2, cell3._signalRelaxationTime);
}

enum class AngleRange
{
    Start, End
};

class SignalTests_BothSides_New
    : public SignalTests_New
    , public testing::WithParamInterface<AngleRange>
{
};

INSTANTIATE_TEST_SUITE_P(SignalTests_BothSides_New, SignalTests_BothSides_New, ::testing::Values(AngleRange::Start, AngleRange::End));

TEST_P(SignalTests_BothSides_New, routeSignalOnRight_sharpMatch)
{
    auto side = GetParam();
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};
    auto data = DataDescription().addCells({
        CellDescription().id(1).pos({0, 0}),
        CellDescription().id(2).pos({1, 0}).signal(signal).signalRoutingRestriction(side == AngleRange::Start ? -45.0f : 45.0f, 90.0f),
        CellDescription().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto cell1 = actualCellById.at(1);
    EXPECT_FALSE(cell1._signal.has_value());
    EXPECT_EQ(0, cell1._signalRelaxationTime);

    auto cell2 = actualCellById.at(2);
    EXPECT_FALSE(cell2._signal.has_value());
    EXPECT_EQ(1, cell2._signalRelaxationTime);

    auto cell3 = actualCellById.at(3);
    EXPECT_TRUE(cell3._signal.has_value());
    EXPECT_TRUE(approxCompare(signal, cell3._signal->_channels));
    EXPECT_EQ(2, cell3._signalRelaxationTime);
}

TEST_P(SignalTests_BothSides_New, routeSignalOnRight_sharpMismatch)
{
    auto side = GetParam();
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};
    auto data = DataDescription().addCells({
        CellDescription().id(1).pos({0, 0}),
        CellDescription().id(2).pos({1, 0}).signal(signal).signalRoutingRestriction(side == AngleRange::Start ? -46.0f : 46.0f, 90.0f),
        CellDescription().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto cell1 = actualCellById.at(1);
    EXPECT_FALSE(cell1._signal.has_value());

    auto cell2 = actualCellById.at(2);
    EXPECT_FALSE(cell2._signal.has_value());

    auto cell3 = actualCellById.at(3);
    EXPECT_FALSE(cell3._signal.has_value());
}