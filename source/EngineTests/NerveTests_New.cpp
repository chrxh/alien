#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class NerveTests_New : public IntegrationTestFramework
{
public:
    NerveTests_New()
        : IntegrationTestFramework()
    {}

    ~NerveTests_New() = default;
};

TEST_F(NerveTests_New, generatePulse_timeBeforeFirstPulse)
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

TEST_F(NerveTests_New, generatePulse_timeAtFirstPulse)
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

TEST_F(NerveTests_New, generatePulse_timeAtSecondPulse)
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

TEST_F(NerveTests_New, generatePulse_timeAfterFirstPulse)
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

TEST_F(NerveTests_New, generatePulse_timeBeforeFirstPulseAlternation)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellFunction(NerveDescription().setPulseMode(97).setAlternationMode(3)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 2);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve = actualCellById.at(1);
    EXPECT_TRUE(nerve.signal.active);
    EXPECT_EQ(1.0f, nerve.signal.channels.at(0));
}

TEST_F(NerveTests_New, generatePulse_timeAtFirstPulseAlternation)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellFunction(NerveDescription().setPulseMode(97).setAlternationMode(3)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 3);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto nerve = actualCellById.at(1);
    EXPECT_TRUE(nerve.signal.active);
    EXPECT_EQ(-1.0f, nerve.signal.channels.at(0));
}

TEST_F(NerveTests_New, generatePulse_timeAtSecondPulseAlternation)
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

TEST_F(NerveTests_New, generatePulse_triangularNetwork)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({0, 0}).setCellFunction(NerveDescription().setPulseMode(10)),
        CellDescription().setId(2).setPos({1, 0}),
        CellDescription().setId(3).setPos({0.5, 0.5}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(20);

    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        auto nerve1 = actualCellById.at(1);
        EXPECT_TRUE(nerve1.signal.active);
        EXPECT_TRUE(approxCompare(1.0f, nerve1.signal.channels.at(0)));
        EXPECT_TRUE(nerve1.signal.prevCellIds.empty());

        auto nerve2 = actualCellById.at(2);
        EXPECT_FALSE(nerve2.signal.active);

        auto nerve3 = actualCellById.at(3);
        EXPECT_FALSE(nerve3.signal.active);
    }
}
