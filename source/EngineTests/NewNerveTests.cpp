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
        CellDescription().setId(1).setCellFunction(NerveDescription().setPulseMode(0)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve = actualCellById.at(1);
    EXPECT_FALSE(nerve.signal.active);
}

TEST_F(NewNerveTests, timeBeforePulse)
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

TEST_F(NewNerveTests, timeDuringPulse)
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

TEST_F(NewNerveTests, timeDuringSecondPulse)
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

TEST_F(NewNerveTests, timeAfterPulse)
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

TEST_F(NewNerveTests, timeDuringPulse_beforeAlternation)
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

TEST_F(NewNerveTests, timeDuringPulse_duringFirstAlternation)
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

TEST_F(NewNerveTests, timeDuringPulse_duringSecondAlternation)
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
