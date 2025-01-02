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

    auto oscillator = actualCellById.at(1);
    EXPECT_FALSE(oscillator.signal.has_value());
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

    auto oscillator1 = actualCellById.at(1);
    EXPECT_FALSE(oscillator1.signal.has_value());

    auto oscillator2 = actualCellById.at(2);
    EXPECT_TRUE(oscillator2.signal.has_value());
    EXPECT_EQ(signal, oscillator2.signal->channels);
    EXPECT_EQ(1, oscillator2.signal->prevCellIds.size());
    EXPECT_EQ(1, oscillator2.signal->prevCellIds[0]);
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

    auto oscillator1 = actualCellById.at(1);
    EXPECT_FALSE(oscillator1.signal.has_value());
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

    auto oscillator1 = actualCellById.at(1);
    EXPECT_FALSE(oscillator1.signal.has_value());
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

    auto oscillator1 = actualCellById.at(1);
    EXPECT_FALSE(oscillator1.signal.has_value());

    auto oscillator2 = actualCellById.at(2);
    EXPECT_TRUE(oscillator2.signal.has_value());
    std::vector<float> sumSignal(signal1.size());
    for (size_t i = 0; i < signal1.size(); ++i) {
        sumSignal[i] = signal1[i] + signal2[i];
    }
    EXPECT_TRUE(approxCompare(sumSignal, oscillator2.signal->channels));
    auto prevCellIdSet = std::set(oscillator2.signal->prevCellIds.begin(), oscillator2.signal->prevCellIds.end());
    EXPECT_EQ((std::set<uint64_t>{1, 3}), prevCellIdSet);

    auto oscillator3 = actualCellById.at(3);
    EXPECT_FALSE(oscillator3.signal.has_value());
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

    auto oscillator1 = actualCellById.at(1);
    EXPECT_TRUE(oscillator1.signal.has_value());
    EXPECT_TRUE(approxCompare(signal, oscillator1.signal->channels));
    EXPECT_EQ(1, oscillator1.signal->prevCellIds.size());
    EXPECT_EQ(2, oscillator1.signal->prevCellIds[0]);

    auto oscillator2 = actualCellById.at(2);
    EXPECT_FALSE(oscillator2.signal.has_value());

    auto oscillator3 = actualCellById.at(3);
    EXPECT_TRUE(oscillator3.signal.has_value());
    EXPECT_TRUE(approxCompare(signal, oscillator3.signal->channels));
    EXPECT_EQ(1, oscillator1.signal->prevCellIds.size());
    EXPECT_EQ(2, oscillator1.signal->prevCellIds[0]);
}
