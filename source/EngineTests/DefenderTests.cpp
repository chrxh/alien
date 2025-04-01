#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionConverterService.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class DefenderTests : public IntegrationTestFramework
{
public:
    static SimulationParameters getParameters()
    {
        SimulationParameters result;
        result.innerFriction = 0;
        result.friction.baseValue = 1;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.defenderAntiAttackerStrength[i] = 1000.0f;
            result.baseValues.radiationType1_strength[i] = 0;
            result.radiationType2_strength[i] = 0;
            for (int j = 0; j < MAX_COLORS; ++j) {
                result.injectorInjectionTime[i][j] = 3;
            }
        }
        return result;
    }
    DefenderTests()
        : IntegrationTestFramework(getParameters())
    {}

    ~DefenderTests() = default;
};

TEST_F(DefenderTests, attackerVsAntiAttacker)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(AttackerDescription()),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription().autoTriggerInterval(1))
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription()
            .id(3)
            .pos({9.0f, 10.0f})
            .cellType(OscillatorDescription()),
        CellDescription()
            .id(4)
            .pos({7.0f, 10.0f})
            .cellType(DefenderDescription().mode(DefenderMode_DefendAgainstAttacker)),
    });
    data.addConnection(1, 2);
    data.addConnection(3, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = getCell(actualData, 1);

    auto origTarget = getCell(data, 3);
    auto actualTarget = getCell(actualData, 3);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(actualAttacker._signal->_channels[0] > NEAR_ZERO);
    EXPECT_LT(origTarget._energy, actualTarget._energy + 0.1f);
}

TEST_F(DefenderTests, attackerVsAntiInjector)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(AttackerDescription()),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription().autoTriggerInterval(1))
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).cellType(OscillatorDescription()),
        CellDescription()
            .id(4)
            .pos({7.0f, 10.0f})
            .cellType(DefenderDescription().mode(DefenderMode_DefendAgainstInjector)),
    });
    data.addConnection(1, 2);
    data.addConnection(3, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = getCell(actualData, 1);

    auto origTarget = getCell(data, 3);
    auto actualTarget = getCell(actualData, 3);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(actualAttacker._signal->_channels[0] > NEAR_ZERO);
    EXPECT_GT(origTarget._energy, actualTarget._energy + 0.1f);
}

TEST_F(DefenderTests, injectorVsAntiAttacker)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(InjectorDescription().mode(InjectorMode_InjectAll).genome(genome)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription().autoTriggerInterval(1))
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).cellType(ConstructorDescription()),
        CellDescription()
            .id(4)
            .pos({7.0f, 10.0f})
            .cellType(DefenderDescription().mode(DefenderMode_DefendAgainstAttacker)),
    });
    data.addConnection(1, 2);
    data.addConnection(3, 4);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 1 + 6 * 3; ++i) {
        _simulationFacade->calcTimesteps(1);
    }

    auto actualData = _simulationFacade->getSimulationData();

    auto actualInjector = getCell(actualData, 1);
    auto actualInjectorFunc = std::get<InjectorDescription>(actualInjector._cellTypeData);

    auto actualTarget = getCell(actualData, 3);
    auto actualTargetFunc = std::get<ConstructorDescription>(actualTarget._cellTypeData);

    auto origInjector = getCell(data, 1);
    auto origInjectorFunc = std::get<InjectorDescription>(origInjector._cellTypeData);

    EXPECT_TRUE(approxCompare(1.0f, actualInjector._signal->_channels[0]));
    EXPECT_EQ(0, actualInjectorFunc._counter);
    EXPECT_EQ(origInjectorFunc._genome, actualTargetFunc._genome);
}

TEST_F(DefenderTests, injectorVsAntiInjector)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(InjectorDescription().mode(InjectorMode_InjectAll).genome(genome)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription().autoTriggerInterval(1))
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).cellType(ConstructorDescription()),
        CellDescription()
            .id(4)
            .pos({7.0f, 10.0f})
            .cellType(DefenderDescription().mode(DefenderMode_DefendAgainstInjector)),
    });
    data.addConnection(1, 2);
    data.addConnection(3, 4);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 1 + 6 * 3; ++i) {
        _simulationFacade->calcTimesteps(1);
    }

    auto actualData = _simulationFacade->getSimulationData();

    auto actualInjector = getCell(actualData, 1);
    auto actualInjectorFunc = std::get<InjectorDescription>(actualInjector._cellTypeData);

    auto origTarget = getCell(data, 3);
    auto origTargetFunc = std::get<ConstructorDescription>(origTarget._cellTypeData);

    auto actualTarget = getCell(actualData, 3);
    auto actualTargetFunc = std::get<ConstructorDescription>(actualTarget._cellTypeData);

    auto origInjector = getCell(data, 1);
    auto origInjectorFunc = std::get<InjectorDescription>(origInjector._cellTypeData);

    EXPECT_TRUE(approxCompare(1.0f, actualInjector._signal->_channels[0]));
    EXPECT_EQ(4, actualInjectorFunc._counter);
    EXPECT_EQ(origTargetFunc._genome, actualTargetFunc._genome);
}
