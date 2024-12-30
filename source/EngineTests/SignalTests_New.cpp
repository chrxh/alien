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
        CellDescription().setId(1),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve = actualCellById.at(1);
    EXPECT_FALSE(nerve.signal.active);
}

TEST_F(SignalTests_New, forwardSignal)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({0, 0}).setSignal(signal),
        CellDescription().setId(2).setPos({1, 0}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve1 = actualCellById.at(1);
    EXPECT_FALSE(nerve1.signal.active);

    auto nerve2 = actualCellById.at(2);
    EXPECT_TRUE(nerve2.signal.active);
    EXPECT_EQ(signal, nerve2.signal.channels);
    EXPECT_EQ(1, nerve2.signal.prevCellIds.size());
    EXPECT_EQ(1, nerve2.signal.prevCellIds[0]);
}

TEST_F(SignalTests_New, vanishSignal_singleCell)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({0, 0}).setSignal(signal),
    });
    
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve1 = actualCellById.at(1);
    EXPECT_FALSE(nerve1.signal.active);
}

TEST_F(SignalTests_New, vanishSignal_withPrevCell)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0};
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({0, 0}).setSignal(SignalDescription().setChannels(signal).setPrevCellIds({2})),
        CellDescription().setId(2).setPos({1, 0}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve1 = actualCellById.at(1);
    EXPECT_FALSE(nerve1.signal.active);
}

TEST_F(SignalTests_New, mergeSignals)
{
    std::vector<float> signal1 = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};
    std::vector<float> signal2 = {-0.5f, -2.0f, 0.5f, 1.0f, 1.5f, -1.5f, 0.5f, -0.5f};
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({0, 0}).setSignal(signal1),
        CellDescription().setId(2).setPos({1, 0}),
        CellDescription().setId(3).setPos({2, 0}).setSignal(signal2),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve1 = actualCellById.at(1);
    EXPECT_FALSE(nerve1.signal.active);

    auto nerve2 = actualCellById.at(2);
    EXPECT_TRUE(nerve2.signal.active);
    std::vector<float> sumSignal(signal1.size());
    for (size_t i = 0; i < signal1.size(); ++i) {
        sumSignal[i] = signal1[i] + signal2[i];
    }
    EXPECT_TRUE(approxCompare(sumSignal, nerve2.signal.channels));
    auto prevCellIdSet = std::set(nerve2.signal.prevCellIds.begin(), nerve2.signal.prevCellIds.end());
    EXPECT_EQ((std::set<uint64_t>{1, 3}), prevCellIdSet);

    auto nerve3 = actualCellById.at(3);
    EXPECT_FALSE(nerve3.signal.active);
}

TEST_F(SignalTests_New, forkSignals)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f};
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({0, 0}),
        CellDescription().setId(2).setPos({1, 0}).setSignal(signal),
        CellDescription().setId(3).setPos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve1 = actualCellById.at(1);
    EXPECT_TRUE(nerve1.signal.active);
    EXPECT_TRUE(approxCompare(signal, nerve1.signal.channels));
    EXPECT_EQ(1, nerve1.signal.prevCellIds.size());
    EXPECT_EQ(2, nerve1.signal.prevCellIds[0]);

    auto nerve2 = actualCellById.at(2);
    EXPECT_FALSE(nerve2.signal.active);

    auto nerve3 = actualCellById.at(3);
    EXPECT_TRUE(nerve3.signal.active);
    EXPECT_TRUE(approxCompare(signal, nerve3.signal.channels));
    EXPECT_EQ(1, nerve1.signal.prevCellIds.size());
    EXPECT_EQ(2, nerve1.signal.prevCellIds[0]);
}
