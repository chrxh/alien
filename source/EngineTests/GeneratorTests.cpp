#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

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
    data._objects = {
        ObjectDescription().id(1).type(CellDescription().cellType(GeneratorDescription().autoTriggerInterval(97))),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getObjectRef(1);
    EXPECT_FALSE(std::get<CellDescription>(generator._type)._signalState == SignalState_Active);
}

TEST_F(GeneratorTests, generatePulse_timeAtFirstPulse)
{
    Description data;
    data._objects = {
        ObjectDescription().id(1).type(CellDescription().cellType(GeneratorDescription().autoTriggerInterval(97))),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(98);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getObjectRef(1);
    ASSERT_TRUE(std::get<CellDescription>(generator._type)._signalState == SignalState_Active);
    EXPECT_EQ(1.0f, std::get<CellDescription>(generator._type)._signal._channels.at(0));
}

TEST_F(GeneratorTests, generatePulse_timeAtFirstPulse_detailedPreview)
{
    Description data;
    data._objects = {
        ObjectDescription().id(1).type(CellDescription().cellType(GeneratorDescription().autoTriggerInterval(97))),
    };

    _simulationFacade->setPreviewData(data);
    _simulationFacade->calcTimestepsForPreview(98, true);
    auto actualData = _simulationFacade->getPreviewData();

    auto generator = actualData.getObjectRef(1);
    ASSERT_TRUE(std::get<CellDescription>(generator._type)._signalState == SignalState_Active);
    EXPECT_EQ(1.0f, std::get<CellDescription>(generator._type)._signal._channels.at(0));
}

TEST_F(GeneratorTests, generatePulse_timeAtSecondPulse)
{
    Description data;
    data._objects = {
        ObjectDescription().id(1).type(CellDescription().cellType(GeneratorDescription().autoTriggerInterval(97 * 2))),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 2 + 1);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getObjectRef(1);
    EXPECT_TRUE(std::get<CellDescription>(generator._type)._signalState == SignalState_Active);
    EXPECT_EQ(1.0f, std::get<CellDescription>(generator._type)._signal._channels.at(0));
}

TEST_F(GeneratorTests, generatePulse_timeAfterFirstPulse)
{
    Description data;
    data._objects = {
        ObjectDescription().id(1).type(CellDescription().cellType(GeneratorDescription().autoTriggerInterval(97))),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(99);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getObjectRef(1);
    EXPECT_FALSE(std::get<CellDescription>(generator._type)._signalState == SignalState_Active);
}

TEST_F(GeneratorTests, generatePulse_timeBeforeFirstPulseAlternation)
{
    Description data;
    data._objects = {
        ObjectDescription().id(1).type(CellDescription().cellType(GeneratorDescription().autoTriggerInterval(97).pulseType(GeneratorPulseType_Alternation).alternationInterval(3))),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 2 + 1);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getObjectRef(1);
    EXPECT_TRUE(std::get<CellDescription>(generator._type)._signalState == SignalState_Active);
    EXPECT_EQ(1.0f, std::get<CellDescription>(generator._type)._signal._channels.at(0));
}

TEST_F(GeneratorTests, generatePulse_timeAtFirstPulseAlternation)
{
    Description data;
    data._objects = {
        ObjectDescription().id(1).type(CellDescription().cellType(GeneratorDescription().autoTriggerInterval(97).pulseType(GeneratorPulseType_Alternation).alternationInterval(3))),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 3 + 1);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getObjectRef(1);
    EXPECT_TRUE(std::get<CellDescription>(generator._type)._signalState == SignalState_Active);
    EXPECT_EQ(-1.0f, std::get<CellDescription>(generator._type)._signal._channels.at(0));
}

TEST_F(GeneratorTests, generatePulse_timeAtSecondPulseAlternation)
{
    Description data;
    data._objects = {
        ObjectDescription().id(1).type(CellDescription().cellType(GeneratorDescription().autoTriggerInterval(97).pulseType(GeneratorPulseType_Alternation).alternationInterval(3))),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(97 * 6 + 1);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getObjectRef(1);
    EXPECT_TRUE(std::get<CellDescription>(generator._type)._signalState == SignalState_Active);
    EXPECT_EQ(1.0f, std::get<CellDescription>(generator._type)._signal._channels.at(0));
}

TEST_F(GeneratorTests, generatePulse_triangularNetwork)
{
    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({0, 0}).type(CellDescription().cellType(GeneratorDescription().autoTriggerInterval(10))),
        ObjectDescription().id(2).pos({1, 0}),
        ObjectDescription().id(3).pos({0.5, 0.5}),
    };
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(20 + 1);

    {
        auto actualData = _simulationFacade->getSimulationData();

        auto generator = actualData.getObjectRef(1);
        EXPECT_TRUE(std::get<CellDescription>(generator._type)._signalState == SignalState_Active);
        EXPECT_TRUE(approxCompare(1.0f, std::get<CellDescription>(generator._type)._signal._channels.at(0)));
        EXPECT_EQ(2, std::get<CellDescription>(generator._type)._signalState);

        auto base1 = actualData.getObjectRef(2);
        EXPECT_FALSE(std::get<CellDescription>(base1._type)._signalState == SignalState_Active);
        EXPECT_EQ(0, std::get<CellDescription>(base1._type)._signalState);

        auto base2 = actualData.getObjectRef(3);
        EXPECT_FALSE(std::get<CellDescription>(base2._type)._signalState == SignalState_Active);
        EXPECT_EQ(0, std::get<CellDescription>(base2._type)._signalState);
    }
}
