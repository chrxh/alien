#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class GeneratorTests : public IntegrationTestFramework
{
public:
    GeneratorTests()
        : IntegrationTestFramework()
    {}

    ~GeneratorTests() = default;
};

TEST_F(GeneratorTests, generatePulse_timeBeforeFirstPulse)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).cellTypeData(GeneratorDescription().autoTriggerInterval(97)),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getCellRef(1);
    EXPECT_FALSE(generator._signal.has_value());
}

TEST_F(GeneratorTests, generatePulse_timeAtFirstPulse)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).cellTypeData(GeneratorDescription().autoTriggerInterval(97)),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(98);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getCellRef(1);
    ASSERT_TRUE(generator._signal.has_value());
    EXPECT_EQ(1.0f, generator._signal->_channels.at(0));
}

TEST_F(GeneratorTests, generatePulse_timeAtSecondPulse)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).cellTypeData(GeneratorDescription().autoTriggerInterval(97 * 2)),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 2 + 1);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getCellRef(1);
    EXPECT_TRUE(generator._signal.has_value());
    EXPECT_EQ(1.0f, generator._signal->_channels.at(0));
}

TEST_F(GeneratorTests, generatePulse_timeAfterFirstPulse)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).cellTypeData(GeneratorDescription().autoTriggerInterval(97)),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(99);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getCellRef(1);
    EXPECT_FALSE(generator._signal.has_value());
}

TEST_F(GeneratorTests, generatePulse_timeBeforeFirstPulseAlternation)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).cellTypeData(GeneratorDescription().autoTriggerInterval(97).pulseType(GeneratorPulseType_Alternation).alternationInterval(3)),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 2 + 1);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getCellRef(1);
    EXPECT_TRUE(generator._signal.has_value());
    EXPECT_EQ(1.0f, generator._signal->_channels.at(0));
}

TEST_F(GeneratorTests, generatePulse_timeAtFirstPulseAlternation)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).cellTypeData(GeneratorDescription().autoTriggerInterval(97).pulseType(GeneratorPulseType_Alternation).alternationInterval(3)),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 3 + 1);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getCellRef(1);
    EXPECT_TRUE(generator._signal.has_value());
    EXPECT_EQ(-1.0f, generator._signal->_channels.at(0));
}

TEST_F(GeneratorTests, generatePulse_timeAtSecondPulseAlternation)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).cellTypeData(GeneratorDescription().autoTriggerInterval(97).pulseType(GeneratorPulseType_Alternation).alternationInterval(3)),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 6 + 1);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getCellRef(1);
    EXPECT_TRUE(generator._signal.has_value());
    EXPECT_EQ(1.0f, generator._signal->_channels.at(0));
}

TEST_F(GeneratorTests, generatePulse_triangularNetwork)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({0, 0}).cellTypeData(GeneratorDescription().autoTriggerInterval(10)),
        CellDescription().id(2).pos({1, 0}),
        CellDescription().id(3).pos({0.5, 0.5}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(20 + 1);

    {
        auto actualData = _simulationFacade->getSimulationData();

        auto generator = actualData.getCellRef(1);
        EXPECT_TRUE(generator._signal.has_value());
        EXPECT_TRUE(approxCompare(1.0f, generator._signal->_channels.at(0)));
        EXPECT_EQ(2, generator._signalRelaxationTime);

        auto base1 = actualData.getCellRef(2);
        EXPECT_FALSE(base1._signal.has_value());
        EXPECT_EQ(0, base1._signalRelaxationTime);

        auto base2 = actualData.getCellRef(3);
        EXPECT_FALSE(base2._signal.has_value());
        EXPECT_EQ(0, base2._signalRelaxationTime);
    }
}
