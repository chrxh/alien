#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class NewNerveTests : public IntegrationTestFramework
{
public:
    NewNerveTests()
        : IntegrationTestFramework()
    {}

    ~NewNerveTests() = default;
};

TEST_F(NewNerveTests, noSignal)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellFunction(NerveDescription()),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve = actualCellById.at(1);
    EXPECT_FALSE(nerve.signal.active);
}

TEST_F(NewNerveTests, forwardInputSignal)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellFunction(NerveDescription()).setSignal({1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0}),
        CellDescription().setId(2).setCellFunction(NerveDescription()),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);

    {
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        auto nerve1 = actualCellById.at(1);
        EXPECT_FALSE(nerve1.signal.active);

        auto nerve2 = actualCellById.at(2);
        EXPECT_TRUE(nerve2.signal.active);
    }
    {
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        auto nerve1 = actualCellById.at(1);
        EXPECT_FALSE(nerve1.signal.active);

        auto nerve2 = actualCellById.at(2);
        EXPECT_FALSE(nerve2.signal.active);
    }
}

TEST_F(NewNerveTests, generatePulse_timeBeforeFirstPulse)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellFunction(NerveDescription().setPulseMode(97)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(96);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve = actualCellById.at(1);
    EXPECT_FALSE(nerve.signal.active);
}

TEST_F(NewNerveTests, generatePulse_timeAtFirstPulse)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellFunction(NerveDescription().setPulseMode(97)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve = actualCellById.at(1);
    EXPECT_TRUE(nerve.signal.active);
    EXPECT_EQ(1.0f, nerve.signal.channels.at(0));
}

TEST_F(NewNerveTests, generatePulse_timeAtSecondPulse)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellFunction(NerveDescription().setPulseMode(97 * 2)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 2);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve = actualCellById.at(1);
    EXPECT_TRUE(nerve.signal.active);
    EXPECT_EQ(1.0f, nerve.signal.channels.at(0));
}

TEST_F(NewNerveTests, generatePulse_timeAfterFirstPulse)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellFunction(NerveDescription().setPulseMode(97)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(98);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve = actualCellById.at(1);
    EXPECT_FALSE(nerve.signal.active);
}

TEST_F(NewNerveTests, generatePulse_timeBeforeFirstPulseAlternation)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellFunction(NerveDescription().setPulseMode(97).setAlternationMode(3)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 2);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve = actualCellById.at(1);
    EXPECT_TRUE(nerve.signal.active);
    EXPECT_EQ(1.0f, nerve.signal.channels.at(0));
}

TEST_F(NewNerveTests, generatePulse_timeAtFirstPulseAlternation)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellFunction(NerveDescription().setPulseMode(97).setAlternationMode(3)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 3);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve = actualCellById.at(1);
    EXPECT_TRUE(nerve.signal.active);
    EXPECT_EQ(-1.0f, nerve.signal.channels.at(0));
}

TEST_F(NewNerveTests, generatePulse_timeAtSecondPulseAlternation)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellFunction(NerveDescription().setPulseMode(97).setAlternationMode(3)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 6);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve = actualCellById.at(1);
    EXPECT_TRUE(nerve.signal.active);
    EXPECT_EQ(1.0f, nerve.signal.channels.at(0));
}
