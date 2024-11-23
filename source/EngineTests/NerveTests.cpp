#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class NerveTests : public IntegrationTestFramework
{
public:
    NerveTests()
        : IntegrationTestFramework()
    {}

    ~NerveTests() = default;
};

TEST_F(NerveTests, noInput_execution)
{
    SignalDescription signal;
    signal.channels = {1, 0, -1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setSignal(signal),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    EXPECT_EQ(SignalDescription(), actualCellById.at(1).signal);
}

TEST_F(NerveTests, noInput_noExecution)
{
    SignalDescription signal;
    signal.channels = {1, 0, -1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(1)
            .setSignal(signal),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    EXPECT_EQ(signal, actualCellById.at(1).signal);
}

TEST_F(NerveTests, inputBlocked)
{
    SignalDescription signal;
    signal.channels = {1, 0, -1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setPos({1.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setSignal(signal),
        CellDescription()
            .setId(2)
            .setPos({2.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(0),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(SignalDescription(), actualCellById.at(1).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(2).signal);
    }
}

TEST_F(NerveTests, outputBlocked)
{
    SignalDescription signal;
    signal.channels = {1, 0, -1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setPos({1.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setOutputBlocked(true)
            .setSignal(signal),
        CellDescription()
            .setId(2)
            .setPos({2.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(SignalDescription(), actualCellById.at(1).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(2).signal);
    }
}

TEST_F(NerveTests, underConstruction1)
{
    SignalDescription signal;
    signal.channels = {1, 0, -1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setPos({1.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setLivingState(LivingState_UnderConstruction)
            .setSignal(signal),
        CellDescription()
            .setId(2)
            .setPos({2.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(SignalDescription(), actualCellById.at(1).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(2).signal);
    }
}

TEST_F(NerveTests, underConstruction2)
{
    SignalDescription signal;
    signal.channels = {1, 0, -1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setPos({1.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setSignal(signal),
        CellDescription()
            .setId(2)
            .setPos({2.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(SignalDescription(), actualCellById.at(1).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(2).signal);
    }
}

TEST_F(NerveTests, transfer)
{
    SignalDescription signal;
    signal.channels = {1, 0, -1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setPos({1.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setSignal(signal),
        CellDescription()
            .setId(2)
            .setPos({2.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5),
        CellDescription()
            .setId(3)
            .setPos({3.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(1)
            .setInputExecutionOrderNumber(0),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(SignalDescription(), actualCellById.at(1).signal);
        EXPECT_EQ(signal, actualCellById.at(2).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(3).signal);
    }

    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(SignalDescription(), actualCellById.at(1).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(2).signal);
        EXPECT_EQ(signal, actualCellById.at(3).signal);
    }
    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(SignalDescription(), actualCellById.at(1).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(2).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(3).signal);
    }
}


TEST_F(NerveTests, cycle)
{
    SignalDescription signal;
    signal.channels = {1, 0, -1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells(
        {CellDescription()
             .setId(1)
             .setPos({1.0f, 1.0f})
             .setCellFunction(NerveDescription())
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(3),
         CellDescription()
             .setId(2)
             .setPos({2.0f, 1.0f})
             .setCellFunction(NerveDescription())
             .setMaxConnections(2)
             .setExecutionOrderNumber(1)
             .setInputExecutionOrderNumber(0),
         CellDescription()
             .setId(3)
             .setPos({2.0f, 2.0f})
             .setCellFunction(NerveDescription())
             .setMaxConnections(2)
             .setExecutionOrderNumber(2)
             .setInputExecutionOrderNumber(1),
         CellDescription()
             .setId(4)
             .setPos({1.0f, 2.0f})
             .setCellFunction(NerveDescription())
             .setMaxConnections(2)
             .setExecutionOrderNumber(3)
             .setInputExecutionOrderNumber(2)
             .setSignal(signal)});
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.addConnection(4, 1);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(signal, actualCellById.at(1).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(2).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(3).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(4).signal);
    }

    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(SignalDescription(), actualCellById.at(1).signal);
        EXPECT_EQ(signal, actualCellById.at(2).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(3).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(4).signal);
    }

    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(SignalDescription(), actualCellById.at(1).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(2).signal);
        EXPECT_EQ(signal, actualCellById.at(3).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(4).signal);
    }

    for (int i = 0; i < 3; ++i) {
        _simulationFacade->calcTimesteps(1);
    }
    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(SignalDescription(), actualCellById.at(1).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(2).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(3).signal);
        EXPECT_EQ(signal, actualCellById.at(4).signal);
    }

    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(signal, actualCellById.at(1).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(2).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(3).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(4).signal);
    }
}

TEST_F(NerveTests, fork)
{
    SignalDescription signal;
    signal.channels = {1, 0, -1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setPos({1.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5),
        CellDescription()
            .setId(2)
            .setPos({2.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setSignal(signal),
        CellDescription()
            .setId(3)
            .setPos({3.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(signal, actualCellById.at(1).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(2).signal);
        EXPECT_EQ(signal, actualCellById.at(3).signal);
    }
}

TEST_F(NerveTests, noFork)
{
    SignalDescription signal;
    signal.channels = {1, 0, -1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({1.0f, 1.0f}).setCellFunction(NerveDescription()).setMaxConnections(2).setExecutionOrderNumber(1),
        CellDescription()
            .setId(2)
            .setPos({2.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setSignal(signal),
        CellDescription()
            .setId(3)
            .setPos({3.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(SignalDescription(), actualCellById.at(1).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(2).signal);
        EXPECT_EQ(signal, actualCellById.at(3).signal);
    }
}

TEST_F(NerveTests, merge)
{
    SignalDescription signal1, signal2, sumSignals;
    signal1.channels = {1, 0, -1, 0, 0, 0, 0, 0};
    signal2.channels = {2, 0, 0.5f, 0, 0, 0, 0, 0};
    sumSignals.channels = {3, 0, -0.5f, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setPos({1.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setSignal(signal1),
        CellDescription()
            .setId(2)
            .setPos({2.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5),
        CellDescription()
            .setId(3)
            .setPos({3.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setSignal(signal2),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(SignalDescription(), actualCellById.at(1).signal);
        EXPECT_EQ(sumSignals, actualCellById.at(2).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(3).signal);
    }
}

TEST_F(NerveTests, sameExecutionOrderNumber)
{
    SignalDescription signal;
    signal.channels = {1, 0, -1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setPos({1.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setSignal(signal),
        CellDescription()
            .setId(2)
            .setPos({2.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(0),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        EXPECT_EQ(SignalDescription(), actualCellById.at(1).signal);
        EXPECT_EQ(SignalDescription(), actualCellById.at(2).signal);
    }
}

TEST_F(NerveTests, constantPulse)
{
    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setPos({1.0f, 1.0f})
            .setCellFunction(NerveDescription().setPulseMode(3).setAlternationMode(0))
            .setMaxConnections(2)
            .setExecutionOrderNumber(0),
         CellDescription()
             .setId(2)
             .setPos({2.0f, 1.0f})
             .setCellFunction(NerveDescription())
             .setMaxConnections(2)
             .setExecutionOrderNumber(1)
             .setInputExecutionOrderNumber(0)
             .setOutputBlocked(true)
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);

    for (int i = 0; i <= 18; ++i) {
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        SignalDescription signal;
        if (i % 18 == 0) {
            signal.channels = {1, 0, 0, 0, 0, 0, 0, 0};
        }
        EXPECT_EQ(signal, actualCellById.at(1).signal);
    }
}

TEST_F(NerveTests, alternatingPulse)
{
    auto data = DataDescription().addCells(
        {CellDescription()
             .setId(1)
             .setPos({1.0f, 1.0f})
             .setCellFunction(NerveDescription().setPulseMode(3).setAlternationMode(4))
             .setMaxConnections(2)
             .setExecutionOrderNumber(0),
         CellDescription()
             .setId(2)
             .setPos({2.0f, 1.0f})
             .setCellFunction(NerveDescription())
             .setMaxConnections(2)
             .setExecutionOrderNumber(1)
             .setInputExecutionOrderNumber(0)
             .setOutputBlocked(true)});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    {
        auto actualCellById = getCellById(_simulationFacade->getSimulationData());
        EXPECT_EQ(SignalDescription().setChannels({1, 0, 0, 0, 0, 0, 0, 0}), actualCellById.at(1).signal);
    }

    for (int pulse = 0; pulse < 3; ++pulse) {
        for (int i = 0; i < 6*3; ++i) {
            _simulationFacade->calcTimesteps(1);
        }
        auto actualCellById = getCellById(_simulationFacade->getSimulationData());
        EXPECT_EQ(SignalDescription().setChannels({1, 0, 0, 0, 0, 0, 0, 0}), actualCellById.at(1).signal);
    }

    for (int pulse = 0; pulse < 4; ++pulse) {
        for (int i = 0; i < 6*3; ++i) {
            _simulationFacade->calcTimesteps(1);
        }
        auto actualCellById = getCellById(_simulationFacade->getSimulationData());
        EXPECT_EQ(SignalDescription().setChannels({-1, 0, 0, 0, 0, 0, 0, 0}), actualCellById.at(1).signal);
    }
    for (int pulse = 0; pulse < 4; ++pulse) {
        for (int i = 0; i < 6*3; ++i) {
            _simulationFacade->calcTimesteps(1);
        }
        auto actualCellById = getCellById(_simulationFacade->getSimulationData());
        EXPECT_EQ(SignalDescription().setChannels({1, 0, 0, 0, 0, 0, 0, 0}), actualCellById.at(1).signal);
    }
}
