#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class OscillatorTests_New : public IntegrationTestFramework
{
public:
    OscillatorTests_New()
        : IntegrationTestFramework()
    {}

    ~OscillatorTests_New() = default;
};

TEST_F(OscillatorTests_New, generatePulse_timeBeforeFirstPulse)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellTypeData(OscillatorDescription().setAutoTriggerInterval(97)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(96);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    EXPECT_FALSE(oscillator.signal.has_value());
}

TEST_F(OscillatorTests_New, generatePulse_timeAtFirstPulse)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellTypeData(OscillatorDescription().setAutoTriggerInterval(97)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(98);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    ASSERT_TRUE(oscillator.signal.has_value());
    EXPECT_EQ(1.0f, oscillator.signal->channels.at(0));
}

TEST_F(OscillatorTests_New, generatePulse_timeAtSecondPulse)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellTypeData(OscillatorDescription().setAutoTriggerInterval(97 * 2)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 2);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    EXPECT_TRUE(oscillator.signal.has_value());
    EXPECT_EQ(1.0f, oscillator.signal->channels.at(0));
}

TEST_F(OscillatorTests_New, generatePulse_timeAfterFirstPulse)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellTypeData(OscillatorDescription().setAutoTriggerInterval(97)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(98);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    EXPECT_FALSE(oscillator.signal.has_value());
}

TEST_F(OscillatorTests_New, generatePulse_timeBeforeFirstPulseAlternation)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellTypeData(OscillatorDescription().setAutoTriggerInterval(97).setAlternationInterval(3)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 2);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    EXPECT_TRUE(oscillator.signal.has_value());
    EXPECT_EQ(1.0f, oscillator.signal->channels.at(0));
}

TEST_F(OscillatorTests_New, generatePulse_timeAtFirstPulseAlternation)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellTypeData(OscillatorDescription().setAutoTriggerInterval(97).setAlternationInterval(3)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 3);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    EXPECT_TRUE(oscillator.signal.has_value());
    EXPECT_EQ(-1.0f, oscillator.signal->channels.at(0));
}

TEST_F(OscillatorTests_New, generatePulse_timeAtSecondPulseAlternation)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setCellTypeData(OscillatorDescription().setAutoTriggerInterval(97).setAlternationInterval(3)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 6);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    EXPECT_TRUE(oscillator.signal.has_value());
    EXPECT_EQ(1.0f, oscillator.signal->channels.at(0));
}

TEST_F(OscillatorTests_New, generatePulse_triangularNetwork)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({0, 0}).setCellTypeData(OscillatorDescription().setAutoTriggerInterval(10)),
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

        auto oscillator = actualCellById.at(1);
        EXPECT_TRUE(oscillator.signal.has_value());
        EXPECT_TRUE(approxCompare(1.0f, oscillator.signal->channels.at(0)));
        EXPECT_EQ(2, oscillator.signalRelaxationTime);

        auto base1 = actualCellById.at(2);
        EXPECT_FALSE(base1.signal.has_value());
        EXPECT_EQ(0, base1.signalRelaxationTime);

        auto base2 = actualCellById.at(3);
        EXPECT_FALSE(base2.signal.has_value());
        EXPECT_EQ(0, base2.signalRelaxationTime);
    }
}
