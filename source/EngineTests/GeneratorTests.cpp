//#include <gtest/gtest.h>
//
//#include <EngineInterface/Description.h>
//#include <EngineInterface/DescriptionEditService.h>
//#include <EngineInterface/SimulationFacade.h>
//
//#include "IntegrationTestFramework.h"
//
//class GeneratorTests : public IntegrationTestFramework
//{
//public:
//    GeneratorTests()
//        : IntegrationTestFramework()
//    {}
//
//    ~GeneratorTests() = default;
//};
//
//TEST_F(GeneratorTests, generatePulse_timeBeforeFirstPulse)
//{
//    auto data = Desc().addCreature(
//        {
//            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().autoTriggerInterval(97))),
//        },
//        CreatureDesc().id(0));
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(97);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto generator = actualData.getObjectRef(1);
//    // Signal should not be active yet (channels should be zero)
//    EXPECT_TRUE(approxCompare(0.0f, generator.getCellRef()._signal._channels.at(0)));
//}
//
//TEST_F(GeneratorTests, generatePulse_timeAtFirstPulse)
//{
//    auto data = Desc().addCreature(
//        {
//            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().autoTriggerInterval(97))),
//        },
//        CreatureDesc().id(0));
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(98);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto generator = actualData.getObjectRef(1);
//    EXPECT_EQ(1.0f, generator.getCellRef()._signal._channels.at(0));
//}
//
//TEST_F(GeneratorTests, generatePulse_timeAtFirstPulse_detailedPreview)
//{
//    auto data = Desc().addCreature(
//        {
//            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().autoTriggerInterval(97))),
//        },
//        CreatureDesc().id(0));
//
//    _simulationFacade->setPreviewData(data);
//    _simulationFacade->calcTimestepsForPreview(98, true);
//    auto actualData = _simulationFacade->getPreviewData();
//
//    auto generator = actualData.getObjectRef(1);
//    EXPECT_EQ(1.0f, generator.getCellRef()._signal._channels.at(0));
//}
//
//TEST_F(GeneratorTests, generatePulse_timeAtSecondPulse)
//{
//    auto data = Desc().addCreature(
//        {
//            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().autoTriggerInterval(97 * 2))),
//        },
//        CreatureDesc().id(0));
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(97 * 2 + 1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto generator = actualData.getObjectRef(1);
//    EXPECT_EQ(1.0f, generator.getCellRef()._signal._channels.at(0));
//}
//
//TEST_F(GeneratorTests, generatePulse_timeAfterFirstPulse)
//{
//    auto data = Desc().addCreature(
//        {
//            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().autoTriggerInterval(97))),
//        },
//        CreatureDesc().id(0));
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(99);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto generator = actualData.getObjectRef(1);
//    // Signal should have been consumed after the pulse
//    EXPECT_TRUE(approxCompare(0.0f, generator.getCellRef()._signal._channels.at(0)));
//}
//
//TEST_F(GeneratorTests, generatePulse_timeBeforeFirstPulseAlternation)
//{
//    auto data = Desc().addCreature(
//        {
//            ObjectDesc().id(1).type(
//                CellDesc().cellType(GeneratorDesc().autoTriggerInterval(97).pulseType(GeneratorPulseType_Alternation).alternationInterval(3))),
//        },
//        CreatureDesc().id(0));
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(97 * 2 + 1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto generator = actualData.getObjectRef(1);
//    EXPECT_EQ(1.0f, generator.getCellRef()._signal._channels.at(0));
//}
//
//TEST_F(GeneratorTests, generatePulse_timeAtFirstPulseAlternation)
//{
//    auto data = Desc().addCreature(
//        {
//            ObjectDesc().id(1).type(
//                CellDesc().cellType(GeneratorDesc().autoTriggerInterval(97).pulseType(GeneratorPulseType_Alternation).alternationInterval(3))),
//        },
//        CreatureDesc().id(0));
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(97 * 3 + 1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto generator = actualData.getObjectRef(1);
//    EXPECT_EQ(-1.0f, generator.getCellRef()._signal._channels.at(0));
//}
//
//TEST_F(GeneratorTests, generatePulse_timeAtSecondPulseAlternation)
//{
//    auto data = Desc().addCreature(
//        {
//            ObjectDesc().id(1).type(
//                CellDesc().cellType(GeneratorDesc().autoTriggerInterval(97).pulseType(GeneratorPulseType_Alternation).alternationInterval(3))),
//        },
//        CreatureDesc().id(0));
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(97 * 6 + 1);
//
//    std::this_thread::sleep_for(std::chrono::milliseconds(1));
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto generator = actualData.getObjectRef(1);
//    EXPECT_EQ(1.0f, generator.getCellRef()._signal._channels.at(0));
//}
//
//TEST_F(GeneratorTests, generatePulse_triangularNetwork)
//{
//    auto data = Desc().addCreature(
//        {
//            ObjectDesc().id(1).pos({0, 0}).type(CellDesc().cellType(GeneratorDesc().autoTriggerInterval(10))),
//            ObjectDesc().id(2).pos({1, 0}),
//            ObjectDesc().id(3).pos({0.5, 0.5}),
//        },
//        CreatureDesc().id(0));
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//    data.addConnection(3, 1);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(20 + 1);
//
//    {
//        auto actualData = _simulationFacade->getSimulationData();
//
//        auto generator = actualData.getObjectRef(1);
//        EXPECT_TRUE(approxCompare(1.0f, generator.getCellRef()._signal._channels.at(0)));
//
//        auto base1 = actualData.getObjectRef(2);
//        EXPECT_TRUE(approxCompare(0.0f, base1.getCellRef()._signal._channels.at(0)));
//
//        auto base2 = actualData.getObjectRef(3);
//        EXPECT_TRUE(approxCompare(0.0f, base2.getCellRef()._signal._channels.at(0)));
//    }
//}
