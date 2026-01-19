//#include <gtest/gtest.h>
//
//#include <EngineInterface/DescriptionEditService.h>
//#include <EngineInterface/Description.h>
//#include <EngineInterface/GenomeDescConverterService.h>
//#include <EngineInterface/SimulationFacade.h>
//#include "IntegrationTestFramework.h"
//
//class DefenderTests : public IntegrationTestFramework
//{
//public:
//    static SimulationParameters getParameters()
//    {
//        SimulationParameters result;
//        result.innerFriction.value = 0;
//        result.friction.baseValue = 1;
//        for (int i = 0; i < MAX_COLORS; ++i) {
//            result.defenderAntiAttackerStrength.value[i] = 1000.0f;
//            result.radiationType1_strength.baseValue[i] = 0;
//            result.radiationType2_strength.value[i] = 0;
//            for (int j = 0; j < MAX_COLORS; ++j) {
//                result.injectorInjectionTime.value[i][j] = 3;
//            }
//        }
//        return result;
//    }
//    DefenderTests()
//        : IntegrationTestFramework(getParameters())
//    {}
//
//    ~DefenderTests() = default;
//};
//
//TEST_F(DefenderTests, attackerVsAntiAttacker)
//{
//    Desc data;
//    data.objects() = {
//        ObjectDesc()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .cellType(AttackerDesc()),
//        ObjectDesc()
//            .id(2)
//            .pos({11.0f, 10.0f})
//            .cellType(GeneratorDesc().autoTriggerInterval(1))
//            .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
//        ObjectDesc()
//            .id(3)
//            .pos({9.0f, 10.0f})
//            .cellType(GeneratorDesc()),
//        ObjectDesc()
//            .id(4)
//            .pos({7.0f, 10.0f})
//            .cellType(DefenderDesc().mode(DefenderMode_DefendAgainstAttacker)),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(3, 4);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//    auto actualAttacker = getCell(actualData, 1);
//
//    auto origTarget = getCell(data, 3);
//    auto actualTarget = getCell(actualData, 3);
//
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//    EXPECT_TRUE(actualAttacker.getCellRef()._signal._channels[0] > NEAR_ZERO);
//    EXPECT_LT(origTarget._energy, actualTarget._energy + 0.1f);
//}
//
//TEST_F(DefenderTests, attackerVsAntiInjector)
//{
//    Desc data;
//    data.objects() = {
//        ObjectDesc()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .cellType(AttackerDesc()),
//        ObjectDesc()
//            .id(2)
//            .pos({11.0f, 10.0f})
//            .cellType(GeneratorDesc().autoTriggerInterval(1))
//            .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
//        ObjectDesc().id(3).pos({9.0f, 10.0f}).type(CellDesc().cellType(GeneratorDesc())),
//        ObjectDesc()
//            .id(4)
//            .pos({7.0f, 10.0f})
//            .cellType(DefenderDesc().mode(DefenderMode_DefendAgainstInjector)),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(3, 4);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//    auto actualAttacker = getCell(actualData, 1);
//
//    auto origTarget = getCell(data, 3);
//    auto actualTarget = getCell(actualData, 3);
//
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//    EXPECT_TRUE(actualAttacker.getCellRef()._signal._channels[0] > NEAR_ZERO);
//    EXPECT_GT(origTarget._energy, actualTarget._energy + 0.1f);
//}
//
//TEST_F(DefenderTests, injectorVsAntiAttacker)
//{
//    auto genome = GenomeDescConverterService::get().convertDescriptionToBytes(GenomeDesc().objects({CellGenomeDesc()}));
//
//    Desc data;
//    data.objects() = {
//        ObjectDesc()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .cellType(InjectorDesc().mode(InjectorMode_InjectAll).genome(genome)),
//        ObjectDesc()
//            .id(2)
//            .pos({11.0f, 10.0f})
//            .cellType(GeneratorDesc().autoTriggerInterval(1))
//            .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
//        ObjectDesc().id(3).pos({9.0f, 10.0f}).type(CellDesc().cellType(ConstructorDesc())),
//        ObjectDesc()
//            .id(4)
//            .pos({7.0f, 10.0f})
//            .cellType(DefenderDesc().mode(DefenderMode_DefendAgainstAttacker)),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(3, 4);
//
//    _simulationFacade->setSimulationData(data);
//    for (int i = 0; i < 1 + 6 * 3; ++i) {
//        _simulationFacade->calcTimesteps(1);
//    }
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto actualInjector = getCell(actualData, 1);
//    auto actualInjectorFunc = std::get<InjectorDesc>(actualInjector.getCellRef()._cellType);
//
//    auto actualTarget = getCell(actualData, 3);
//    auto actualTargetFunc = std::get<ConstructorDesc>(actualTarget.getCellRef()._cellType);
//
//    auto origInjector = getCell(data, 1);
//    auto origInjectorFunc = std::get<InjectorDesc>(origInjector.getCellRef()._cellType);
//
//    EXPECT_TRUE(approxCompare(1.0f, actualInjector.getCellRef()._signal._channels[0]));
//    EXPECT_EQ(0, actualInjectorFunc._counter);
//    EXPECT_EQ(origInjectorFunc._genome, actualTargetFunc._genome);
//}
//
//TEST_F(DefenderTests, injectorVsAntiInjector)
//{
//    auto genome = GenomeDescConverterService::get().convertDescriptionToBytes(GenomeDesc().objects({CellGenomeDesc()}));
//
//    Desc data;
//    data.objects() = {
//        ObjectDesc()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .cellType(InjectorDesc().mode(InjectorMode_InjectAll).genome(genome)),
//        ObjectDesc()
//            .id(2)
//            .pos({11.0f, 10.0f})
//            .cellType(GeneratorDesc().autoTriggerInterval(1))
//            .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
//        ObjectDesc().id(3).pos({9.0f, 10.0f}).type(CellDesc().cellType(ConstructorDesc())),
//        ObjectDesc()
//            .id(4)
//            .pos({7.0f, 10.0f})
//            .cellType(DefenderDesc().mode(DefenderMode_DefendAgainstInjector)),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(3, 4);
//
//    _simulationFacade->setSimulationData(data);
//    for (int i = 0; i < 1 + 6 * 3; ++i) {
//        _simulationFacade->calcTimesteps(1);
//    }
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto actualInjector = getCell(actualData, 1);
//    auto actualInjectorFunc = std::get<InjectorDesc>(actualInjector.getCellRef()._cellType);
//
//    auto origTarget = getCell(data, 3);
//    auto origTargetFunc = std::get<ConstructorDesc>(origTarget.getCellRef()._cellType);
//
//    auto actualTarget = getCell(actualData, 3);
//    auto actualTargetFunc = std::get<ConstructorDesc>(actualTarget.getCellRef()._cellType);
//
//    auto origInjector = getCell(data, 1);
//    auto origInjectorFunc = std::get<InjectorDesc>(origInjector.getCellRef()._cellType);
//
//    EXPECT_TRUE(approxCompare(1.0f, actualInjector.getCellRef()._signal._channels[0]));
//    EXPECT_EQ(4, actualInjectorFunc._counter);
//    EXPECT_EQ(origTargetFunc._genome, actualTargetFunc._genome);
//}
