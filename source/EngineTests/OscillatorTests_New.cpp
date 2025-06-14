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
    auto data = CollectionDescription().addCells({
        CellDescription().id(1).cellTypeData(OscillatorDescription().autoTriggerInterval(97)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    EXPECT_FALSE(oscillator._signal.has_value());
}

TEST_F(OscillatorTests_New, generatePulse_timeAtFirstPulse)
{
    auto data = CollectionDescription().addCells({
        CellDescription().id(1).cellTypeData(OscillatorDescription().autoTriggerInterval(97)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(98);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    ASSERT_TRUE(oscillator._signal.has_value());
    EXPECT_EQ(1.0f, oscillator._signal->_channels.at(0));
}

TEST_F(OscillatorTests_New, generatePulse_timeAtSecondPulse)
{
    auto data = CollectionDescription().addCells({
        CellDescription().id(1).cellTypeData(OscillatorDescription().autoTriggerInterval(97 * 2)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 2 + 1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    EXPECT_TRUE(oscillator._signal.has_value());
    EXPECT_EQ(1.0f, oscillator._signal->_channels.at(0));
}

TEST_F(OscillatorTests_New, generatePulse_timeAfterFirstPulse)
{
    auto data = CollectionDescription().addCells({
        CellDescription().id(1).cellTypeData(OscillatorDescription().autoTriggerInterval(97)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(99);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    EXPECT_FALSE(oscillator._signal.has_value());
}

TEST_F(OscillatorTests_New, generatePulse_timeBeforeFirstPulseAlternation)
{
    auto data = CollectionDescription().addCells({
        CellDescription().id(1).cellTypeData(OscillatorDescription().autoTriggerInterval(97).pulseType(OscillatorPulseType_Alternation).alternationInterval(3)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 2 + 1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    EXPECT_TRUE(oscillator._signal.has_value());
    EXPECT_EQ(1.0f, oscillator._signal->_channels.at(0));
}

TEST_F(OscillatorTests_New, generatePulse_timeAtFirstPulseAlternation)
{
    auto data = CollectionDescription().addCells({
        CellDescription().id(1).cellTypeData(OscillatorDescription().autoTriggerInterval(97).pulseType(OscillatorPulseType_Alternation).alternationInterval(3)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 3 + 1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    EXPECT_TRUE(oscillator._signal.has_value());
    EXPECT_EQ(-1.0f, oscillator._signal->_channels.at(0));
}

TEST_F(OscillatorTests_New, generatePulse_timeAtSecondPulseAlternation)
{
    auto data = CollectionDescription().addCells({
        CellDescription().id(1).cellTypeData(OscillatorDescription().autoTriggerInterval(97).pulseType(OscillatorPulseType_Alternation).alternationInterval(3)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 6 + 1);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto oscillator = actualCellById.at(1);
    EXPECT_TRUE(oscillator._signal.has_value());
    EXPECT_EQ(1.0f, oscillator._signal->_channels.at(0));
}

TEST_F(OscillatorTests_New, generatePulse_triangularNetwork)
{
    auto data = CollectionDescription().addCells({
        CellDescription().id(1).pos({0, 0}).cellTypeData(OscillatorDescription().autoTriggerInterval(10)),
        CellDescription().id(2).pos({1, 0}),
        CellDescription().id(3).pos({0.5, 0.5}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(20 + 1);

    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCellById = getCellById(actualData);

        auto oscillator = actualCellById.at(1);
        EXPECT_TRUE(oscillator._signal.has_value());
        EXPECT_TRUE(approxCompare(1.0f, oscillator._signal->_channels.at(0)));
        EXPECT_EQ(2, oscillator._signalRelaxationTime);

        auto base1 = actualCellById.at(2);
        EXPECT_FALSE(base1._signal.has_value());
        EXPECT_EQ(0, base1._signalRelaxationTime);

        auto base2 = actualCellById.at(3);
        EXPECT_FALSE(base2._signal.has_value());
        EXPECT_EQ(0, base2._signalRelaxationTime);
    }
}
