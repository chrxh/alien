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
        result.baseValues.friction = 1;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.cellTypeDefenderAgainstAttackerStrength[i] = 1000.0f;
            result.baseValues.radiationCellAgeStrength[i] = 0;
            result.highRadiationFactor[i] = 0;
            for (int j = 0; j < MAX_COLORS; ++j) {
                result.cellTypeInjectorDurationColorMatrix[i][j] = 3;
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
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setCellTypeData(AttackerDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellTypeData(OscillatorDescription().setPulseMode(1))
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription()
            .setId(3)
            .setPos({9.0f, 10.0f})
            .setCellTypeData(OscillatorDescription()),
        CellDescription()
            .setId(4)
            .setPos({7.0f, 10.0f})
            .setCellTypeData(DefenderDescription().setMode(DefenderMode_DefendAgainstAttacker)),
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
    EXPECT_TRUE(actualAttacker.signal->channels[0] > NEAR_ZERO);
    EXPECT_LT(origTarget.energy, actualTarget.energy + 0.1f);
}

TEST_F(DefenderTests, attackerVsAntiInjector)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setCellTypeData(AttackerDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellTypeData(OscillatorDescription().setPulseMode(1))
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setCellTypeData(OscillatorDescription()),
        CellDescription()
            .setId(4)
            .setPos({7.0f, 10.0f})
            .setCellTypeData(DefenderDescription().setMode(DefenderMode_DefendAgainstInjector)),
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
    EXPECT_TRUE(actualAttacker.signal->channels[0] > NEAR_ZERO);
    EXPECT_GT(origTarget.energy, actualTarget.energy + 0.1f);
}

TEST_F(DefenderTests, injectorVsAntiAttacker)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setCellTypeData(InjectorDescription().setMode(InjectorMode_InjectAll).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellTypeData(OscillatorDescription().setPulseMode(1))
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setCellTypeData(ConstructorDescription()),
        CellDescription()
            .setId(4)
            .setPos({7.0f, 10.0f})
            .setCellTypeData(DefenderDescription().setMode(DefenderMode_DefendAgainstAttacker)),
    });
    data.addConnection(1, 2);
    data.addConnection(3, 4);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 1 + 6 * 3; ++i) {
        _simulationFacade->calcTimesteps(1);
    }

    auto actualData = _simulationFacade->getSimulationData();

    auto actualInjector = getCell(actualData, 1);
    auto actualInjectorFunc = std::get<InjectorDescription>(actualInjector.cellTypeData);

    auto actualTarget = getCell(actualData, 3);
    auto actualTargetFunc = std::get<ConstructorDescription>(actualTarget.cellTypeData);

    auto origInjector = getCell(data, 1);
    auto origInjectorFunc = std::get<InjectorDescription>(origInjector.cellTypeData);

    EXPECT_TRUE(approxCompare(1.0f, actualInjector.signal->channels[0]));
    EXPECT_EQ(0, actualInjectorFunc.counter);
    EXPECT_EQ(origInjectorFunc.genome, actualTargetFunc.genome);
}

TEST_F(DefenderTests, injectorVsAntiInjector)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setCellTypeData(InjectorDescription().setMode(InjectorMode_InjectAll).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellTypeData(OscillatorDescription().setPulseMode(1))
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setCellTypeData(ConstructorDescription()),
        CellDescription()
            .setId(4)
            .setPos({7.0f, 10.0f})
            .setCellTypeData(DefenderDescription().setMode(DefenderMode_DefendAgainstInjector)),
    });
    data.addConnection(1, 2);
    data.addConnection(3, 4);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 1 + 6 * 3; ++i) {
        _simulationFacade->calcTimesteps(1);
    }

    auto actualData = _simulationFacade->getSimulationData();

    auto actualInjector = getCell(actualData, 1);
    auto actualInjectorFunc = std::get<InjectorDescription>(actualInjector.cellTypeData);

    auto origTarget = getCell(data, 3);
    auto origTargetFunc = std::get<ConstructorDescription>(origTarget.cellTypeData);

    auto actualTarget = getCell(actualData, 3);
    auto actualTargetFunc = std::get<ConstructorDescription>(actualTarget.cellTypeData);

    auto origInjector = getCell(data, 1);
    auto origInjectorFunc = std::get<InjectorDescription>(origInjector.cellTypeData);

    EXPECT_TRUE(approxCompare(1.0f, actualInjector.signal->channels[0]));
    EXPECT_EQ(4, actualInjectorFunc.counter);
    EXPECT_EQ(origTargetFunc.genome, actualTargetFunc.genome);
}
