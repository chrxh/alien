#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class DefenderTests : public IntegrationTestFramework
{
public:
    DefenderTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
            _parameters.attackerEnergyCost.baseValue[i] = 0;
            _parameters.attackerStrength.value[i] = 0.5f;
            _parameters.attackerRadius.value[i] = 3.5f;
            _parameters.defenderAntiAttackerStrength.value[i] = 1000.0f;
            _parameters.injectorEnergyCost.value[i] = 0;
            _parameters.injectorRadius.value[i] = 3.5f;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~DefenderTests() = default;
};

/**
 * Test: attackerVsAntiAttacker
 * An attacker attacks a target creature that has a defender cell in DefendAgainstAttacker mode.
 * The attack should be heavily reduced (defenderAntiAttackerStrength=1000 → divisor=1001).
 */
TEST_F(DefenderTests, attackerVsAntiAttacker)
{
    NeuralNetworkDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    SensorLastMatchDesc lastMatch;
    lastMatch._creatureId = 2;
    lastMatch._pos = {100.0f, 103.0f};

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(AttackerDesc().mode(AttackCreatureDesc())).neuralNetwork(nn)),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().cellType(SensorDesc().autoTrigger(false).lastMatch(lastMatch))),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);

    // Target creature with anti-attacker defender connected
    data.addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 103.0f}).type(CellDesc().usableEnergy(100.0f)),
            ObjectDesc().id(101).pos({101.0f, 103.0f}).type(CellDesc().cellType(DefenderDesc().mode(DefenderMode_DefendAgainstAttacker))),
        },
        CreatureDesc().id(2));
    data.addConnection(100, 101);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Attack should be blocked by anti-attacker defender (strength=1000 → divisor=1001)
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

/**
 * Test: attackerVsAntiInjector
 * An attacker attacks a target creature that has a defender cell in DefendAgainstInjector mode.
 * The wrong defender mode should not block the attack, so energy should be drained.
 */
TEST_F(DefenderTests, attackerVsAntiInjector)
{
    NeuralNetworkDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    SensorLastMatchDesc lastMatch;
    lastMatch._creatureId = 2;
    lastMatch._pos = {100.0f, 103.0f};

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(AttackerDesc().mode(AttackCreatureDesc())).neuralNetwork(nn)),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().cellType(SensorDesc().autoTrigger(false).lastMatch(lastMatch))),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);

    // Target creature with anti-injector defender connected (wrong mode for blocking attacker)
    data.addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 103.0f}).type(CellDesc().usableEnergy(100.0f)),
            ObjectDesc().id(101).pos({101.0f, 103.0f}).type(CellDesc().cellType(DefenderDesc().mode(DefenderMode_DefendAgainstInjector))),
        },
        CreatureDesc().id(2));
    data.addConnection(100, 101);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Attack should succeed because defender is in anti-injector mode (not anti-attacker)
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - 1.0f);
}

/**
 * Test: injectorVsAntiAttacker
 * An injector injects a target creature that has a defender cell in DefendAgainstAttacker mode.
 * The wrong defender mode should not block the injection, so it should succeed.
 */
TEST_F(DefenderTests, injectorVsAntiAttacker)
{
    auto injectorGenome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc().color(1)}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().neuralNetwork(NeuralNetworkDesc().bias(0, 1.0f)).cellType(InjectorDesc().geneIndex(2))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(1),
        injectorGenome);
    data.addConnection(1, 2);

    // Target creature with constructor and anti-attacker defender (wrong mode for blocking injector)
    auto targetGenome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().color(2)}),
    });
    data.addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 103.0f}).type(CellDesc().usableEnergy(100.0f).constructor(ConstructorDesc())),
            ObjectDesc().id(101).pos({101.0f, 103.0f}).type(CellDesc().cellType(DefenderDesc().mode(DefenderMode_DefendAgainstAttacker))),
        },
        CreatureDesc().id(2),
        targetGenome);
    data.addConnection(100, 101);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualInjector = actualData.getObjectRef(1);
    auto actualTargetConstructor = actualData.getObjectRef(100).getCellRef()._constructor.value();

    // Injection should succeed because defender is in anti-attacker mode (not anti-injector)
    EXPECT_TRUE(approxCompare(1.0f, actualInjector.getCellRef()._signal._channels[Channels::InjectorSuccess]));
    EXPECT_EQ(2, actualTargetConstructor._geneIndex);

    // Verify the injector's genome was injected into the target creature
    auto actualTargetCreatureId = actualData.getObjectRef(100).getCellRef()._creatureId;
    auto actualTargetCreature = actualData.getCreatureRef(actualTargetCreatureId);
    auto actualTargetGenome = actualData.getGenomeRef(actualTargetCreature._genomeId);
    EXPECT_EQ(injectorGenome._genes, actualTargetGenome._genes);
}

/**
 * Test: injectorVsAntiInjector
 * An injector injects a target creature that has a defender cell in DefendAgainstInjector mode.
 * The defender increases injection cost by 1.5x, making injection fail when the injector can't afford it.
 */
TEST_F(DefenderTests, injectorVsAntiInjector)
{
    // Set injection cost so that with defender the injector can't afford it
    // usableEnergy=100, minCellEnergy=50, cost=40: without defender 100-40=60>50 OK; with defender 100-60=40<50 FAIL
    for (int i = 0; i < MAX_COLORS; ++i) {
        _parameters.injectorEnergyCost.value[i] = 40.0f;
    }
    _simulationFacade->setSimulationParameters(_parameters);

    auto injectorGenome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc().color(1)}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().neuralNetwork(NeuralNetworkDesc().bias(0, 1.0f)).cellType(InjectorDesc().geneIndex(2))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(1),
        injectorGenome);
    data.addConnection(1, 2);

    // Target creature with constructor and anti-injector defender (correct mode)
    auto targetGenome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().color(2)}),
    });
    data.addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 103.0f}).type(CellDesc().usableEnergy(100.0f).constructor(ConstructorDesc().geneIndex(0))),
            ObjectDesc().id(101).pos({101.0f, 103.0f}).type(CellDesc().cellType(DefenderDesc().mode(DefenderMode_DefendAgainstInjector))),
        },
        CreatureDesc().id(2),
        targetGenome);
    data.addConnection(100, 101);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualInjector = actualData.getObjectRef(1);
    auto actualConstructor = actualData.getObjectRef(100).getCellRef()._constructor.value();

    // Injection should fail because defender increases cost beyond what the injector can afford
    EXPECT_TRUE(approxCompare(0.0f, actualInjector.getCellRef()._signal._channels[Channels::InjectorSuccess]));
    EXPECT_EQ(0, actualConstructor._geneIndex);

    // Verify the injector's genome was NOT injected — target still has original genome
    auto actualTargetCreatureId = actualData.getObjectRef(100).getCellRef()._creatureId;
    auto actualTargetCreature = actualData.getCreatureRef(actualTargetCreatureId);
    auto actualTargetGenome = actualData.getGenomeRef(actualTargetCreature._genomeId);
    EXPECT_EQ(targetGenome._genes, actualTargetGenome._genes);
}
