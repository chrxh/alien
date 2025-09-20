#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Description.h"
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
    Description data;
    data._cells = {
        CellDescription().id(1).cellType(GeneratorDescription().autoTriggerInterval(97)),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getCellRef(1);
    EXPECT_FALSE(generator._signal.has_value());
}

TEST_F(GeneratorTests, generatePulse_timeAtFirstPulse)
{
    Description data;
    data._cells = {
        CellDescription().id(1).cellType(GeneratorDescription().autoTriggerInterval(97)),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(98);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getCellRef(1);
    ASSERT_TRUE(generator._signal.has_value());
    EXPECT_EQ(1.0f, generator._signal->_channels.at(0));
}

TEST_F(GeneratorTests, generatePulse_timeAtFirstPulse_detailedPreview)
{
    Description data;
    data._cells = {
        CellDescription().id(1).cellType(GeneratorDescription().autoTriggerInterval(97)),
    };

    _simulationFacade->setPreviewData(data);
    _simulationFacade->calcTimestepsForPreview(98, true);
    auto actualData = _simulationFacade->getPreviewData();

    auto generator = actualData.getCellRef(1);
    ASSERT_TRUE(generator._signal.has_value());
    EXPECT_EQ(1.0f, generator._signal->_channels.at(0));
}

TEST_F(GeneratorTests, generatePulse_timeAtSecondPulse)
{
    Description data;
    data._cells = {
        CellDescription().id(1).cellType(GeneratorDescription().autoTriggerInterval(97 * 2)),
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
    Description data;
    data._cells = {
        CellDescription().id(1).cellType(GeneratorDescription().autoTriggerInterval(97)),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(99);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getCellRef(1);
    EXPECT_FALSE(generator._signal.has_value());
}

TEST_F(GeneratorTests, generatePulse_timeBeforeFirstPulseAlternation)
{
    Description data;
    data._cells = {
        CellDescription().id(1).cellType(GeneratorDescription().autoTriggerInterval(97).pulseType(GeneratorPulseType_Alternation).alternationInterval(3)),
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
    Description data;
    data._cells = {
        CellDescription().id(1).cellType(GeneratorDescription().autoTriggerInterval(97).pulseType(GeneratorPulseType_Alternation).alternationInterval(3)),
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
    Description data;
    data._cells = {
        CellDescription().id(1).cellType(GeneratorDescription().autoTriggerInterval(97).pulseType(GeneratorPulseType_Alternation).alternationInterval(3)),
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
    Description data;
    data._cells = {
        CellDescription().id(1).pos({0, 0}).cellType(GeneratorDescription().autoTriggerInterval(10)),
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
        EXPECT_EQ(2, generator._signalState);

        auto base1 = actualData.getCellRef(2);
        EXPECT_FALSE(base1._signal.has_value());
        EXPECT_EQ(0, base1._signalState);

        auto base2 = actualData.getCellRef(3);
        EXPECT_FALSE(base2._signal.has_value());
        EXPECT_EQ(0, base2._signalState);
    }
}
