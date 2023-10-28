#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "EngineInterface/SimulationController.h"
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
            result.cellFunctionDefenderAgainstAttackerStrength[i] = 1000.0f;
            result.baseValues.radiationCellAgeStrength[i] = 0;
            result.highRadiationFactor[i] = 0;
            for (int j = 0; j < MAX_COLORS; ++j) {
                result.cellFunctionInjectorDurationColorMatrix[i][j] = 3;
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
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(AttackerDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription().setPulseMode(1))
            .setActivity({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription()
            .setId(3)
            .setPos({9.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(NerveDescription()),
        CellDescription()
            .setId(4)
            .setPos({7.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(DefenderDescription().setMode(DefenderMode_DefendAgainstAttacker)),
    });
    data.addConnection(1, 2);
    data.addConnection(3, 4);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualAttacker = getCell(actualData, 1);

    auto origTarget = getCell(data, 3);
    auto actualTarget = getCell(actualData, 3);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(actualAttacker.activity.channels[0] > NEAR_ZERO);
    EXPECT_LT(origTarget.energy, actualTarget.energy + 0.1f);
}

TEST_F(DefenderTests, attackerVsAntiInjector)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(AttackerDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription().setPulseMode(1))
            .setActivity({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMaxConnections(2).setExecutionOrderNumber(0).setCellFunction(NerveDescription()),
        CellDescription()
            .setId(4)
            .setPos({7.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(DefenderDescription().setMode(DefenderMode_DefendAgainstInjector)),
    });
    data.addConnection(1, 2);
    data.addConnection(3, 4);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualAttacker = getCell(actualData, 1);

    auto origTarget = getCell(data, 3);
    auto actualTarget = getCell(actualData, 3);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(actualAttacker.activity.channels[0] > NEAR_ZERO);
    EXPECT_GT(origTarget.energy, actualTarget.energy + 0.1f);
}

TEST_F(DefenderTests, injectorVsAntiAttacker)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(InjectorDescription().setMode(InjectorMode_InjectAll).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription().setPulseMode(1))
            .setActivity({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMaxConnections(2).setExecutionOrderNumber(0).setCellFunction(ConstructorDescription()),
        CellDescription()
            .setId(4)
            .setPos({7.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(DefenderDescription().setMode(DefenderMode_DefendAgainstAttacker)),
    });
    data.addConnection(1, 2);
    data.addConnection(3, 4);

    _simController->setSimulationData(data);
    for (int i = 0; i < 1 + 6 * 3; ++i) {
        _simController->calcTimesteps(1);
    }

    auto actualData = _simController->getSimulationData();

    auto actualInjector = getCell(actualData, 1);
    auto actualInjectorFunc = std::get<InjectorDescription>(*actualInjector.cellFunction);

    auto actualTarget = getCell(actualData, 3);
    auto actualTargetFunc = std::get<ConstructorDescription>(*actualTarget.cellFunction);

    auto origInjector = getCell(data, 1);
    auto origInjectorFunc = std::get<InjectorDescription>(*origInjector.cellFunction);

    EXPECT_TRUE(approxCompare(1.0f, actualInjector.activity.channels[0]));
    EXPECT_EQ(0, actualInjectorFunc.counter);
    EXPECT_EQ(origInjectorFunc.genome, actualTargetFunc.genome);
}

TEST_F(DefenderTests, injectorVsAntiInjector)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(InjectorDescription().setMode(InjectorMode_InjectAll).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription().setPulseMode(1))
            .setActivity({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMaxConnections(2).setExecutionOrderNumber(0).setCellFunction(ConstructorDescription()),
        CellDescription()
            .setId(4)
            .setPos({7.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(DefenderDescription().setMode(DefenderMode_DefendAgainstInjector)),
    });
    data.addConnection(1, 2);
    data.addConnection(3, 4);

    _simController->setSimulationData(data);
    for (int i = 0; i < 1 + 6 * 3; ++i) {
        _simController->calcTimesteps(1);
    }

    auto actualData = _simController->getSimulationData();

    auto actualInjector = getCell(actualData, 1);
    auto actualInjectorFunc = std::get<InjectorDescription>(*actualInjector.cellFunction);

    auto origTarget = getCell(data, 3);
    auto origTargetFunc = std::get<ConstructorDescription>(*origTarget.cellFunction);

    auto actualTarget = getCell(actualData, 3);
    auto actualTargetFunc = std::get<ConstructorDescription>(*actualTarget.cellFunction);

    auto origInjector = getCell(data, 1);
    auto origInjectorFunc = std::get<InjectorDescription>(*origInjector.cellFunction);

    EXPECT_TRUE(approxCompare(1.0f, actualInjector.activity.channels[0]));
    EXPECT_EQ(4, actualInjectorFunc.counter);
    EXPECT_EQ(origTargetFunc.genome, actualTargetFunc.genome);
}
