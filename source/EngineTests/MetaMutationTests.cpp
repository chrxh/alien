#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class MetaMutationTests : public MutationTestsBase
{
};

TEST_F(MetaMutationTests, metaMutation_neuronRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().nodeProbability(0.5f).weightChangeSigma(0.5f).biasChangeSigma(0.5f).actfnChangeProbability(0.5f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().nodeProbability(0.5f).weightChangeSigma(0.5f).biasChangeSigma(0.5f).actfnChangeProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.neuronsMetaMutationsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    bool anyChanged = actualGenome._mutationRates._neuronMutations[0]._nodeProbability != 0.5f
        || actualGenome._mutationRates._neuronMutations[0]._weightChangeSigma != 0.5f || actualGenome._mutationRates._neuronMutations[0]._biasChangeSigma != 0.5f
        || actualGenome._mutationRates._neuronMutations[0]._actfnChangeProbability != 0.5f
        || actualGenome._mutationRates._neuronMutations[1]._nodeProbability != 0.5f || actualGenome._mutationRates._neuronMutations[1]._weightChangeSigma != 0.5f
        || actualGenome._mutationRates._neuronMutations[1]._biasChangeSigma != 0.5f
        || actualGenome._mutationRates._neuronMutations[1]._actfnChangeProbability != 0.5f;
    EXPECT_TRUE(anyChanged);

    EXPECT_GE(actualGenome._mutationRates._neuronMutations[0]._nodeProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[0]._weightChangeSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[0]._biasChangeSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[0]._actfnChangeProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[1]._nodeProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[1]._weightChangeSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[1]._biasChangeSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[1]._actfnChangeProbability, 0.0f);
}

TEST_F(MetaMutationTests, metaMutation_neuronRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().nodeProbability(0.5f).weightChangeSigma(0.5f).biasChangeSigma(0.5f).actfnChangeProbability(0.5f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().nodeProbability(0.5f).weightChangeSigma(0.5f).biasChangeSigma(0.5f).actfnChangeProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.neuronsMetaMutationsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._nodeProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._weightChangeSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._biasChangeSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._actfnChangeProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._nodeProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._weightChangeSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._biasChangeSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._actfnChangeProbability, 0.5f);
}

TEST_F(MetaMutationTests, metaMutation_connectionRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().nodeProbability(0.5f).valueChangeSigma(0.5f);
    genome._mutationRates._connectionMutations[1] = ConnectionMutationDesc().nodeProbability(0.5f).valueChangeSigma(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.connectionsMetaMutationsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    bool anyChanged = actualGenome._mutationRates._connectionMutations[0]._nodeProbability != 0.5f
        || actualGenome._mutationRates._connectionMutations[0]._valueChangeSigma != 0.5f || actualGenome._mutationRates._connectionMutations[1]._nodeProbability != 0.5f
        || actualGenome._mutationRates._connectionMutations[1]._valueChangeSigma != 0.5f;
    EXPECT_TRUE(anyChanged);
    EXPECT_GE(actualGenome._mutationRates._connectionMutations[0]._nodeProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._connectionMutations[0]._valueChangeSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._connectionMutations[1]._nodeProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._connectionMutations[1]._valueChangeSigma, 0.0f);
}

TEST_F(MetaMutationTests, metaMutation_connectionRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().nodeProbability(0.5f).valueChangeSigma(0.5f);
    genome._mutationRates._connectionMutations[1] = ConnectionMutationDesc().nodeProbability(0.5f).valueChangeSigma(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.connectionsMetaMutationsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_EQ(actualGenome._mutationRates._connectionMutations[0]._nodeProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._connectionMutations[0]._valueChangeSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._connectionMutations[1]._nodeProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._connectionMutations[1]._valueChangeSigma, 0.5f);
}

TEST_F(MetaMutationTests, metaMutation_cellTypePropertyRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().nodeProbability(0.5f).valueChangeSigma(0.5f).enumChangeProbability(0.5f);
    genome._mutationRates._cellTypePropertiesMutations[1] = CellTypePropertiesMutationDesc().nodeProbability(0.4f).valueChangeSigma(0.4f).enumChangeProbability(0.4f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.cellTypePropertiesMetaMutationsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    bool anyChanged = actualGenome._mutationRates._cellTypePropertiesMutations[0]._nodeProbability != 0.5f
        || actualGenome._mutationRates._cellTypePropertiesMutations[0]._valueChangeSigma != 0.5f
        || actualGenome._mutationRates._cellTypePropertiesMutations[0]._enumChangeProbability != 0.5f
        || actualGenome._mutationRates._cellTypePropertiesMutations[1]._nodeProbability != 0.4f
        || actualGenome._mutationRates._cellTypePropertiesMutations[1]._valueChangeSigma != 0.4f
        || actualGenome._mutationRates._cellTypePropertiesMutations[1]._enumChangeProbability != 0.4f;
    EXPECT_TRUE(anyChanged);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[0]._nodeProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[0]._valueChangeSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[0]._enumChangeProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[1]._nodeProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[1]._valueChangeSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[1]._enumChangeProbability, 0.0f);
}

TEST_F(MetaMutationTests, metaMutation_cellTypePropertyRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().nodeProbability(0.5f).valueChangeSigma(0.5f).enumChangeProbability(0.5f);
    genome._mutationRates._cellTypePropertiesMutations[1] = CellTypePropertiesMutationDesc().nodeProbability(0.4f).valueChangeSigma(0.4f).enumChangeProbability(0.4f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.cellTypePropertiesMetaMutationsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[0]._nodeProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[0]._valueChangeSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[0]._enumChangeProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[1]._nodeProbability, 0.4f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[1]._valueChangeSigma, 0.4f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[1]._enumChangeProbability, 0.4f);
}

TEST_F(MetaMutationTests, metaMutation_cellTypeModeRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypeModeMutation = CellTypeModeMutationDesc().nodeProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.cellTypeModeMetaMutationsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_NE(actualGenome._mutationRates._cellTypeModeMutation._nodeProbability, 0.5f);
    EXPECT_GE(actualGenome._mutationRates._cellTypeModeMutation._nodeProbability, 0.0f);
    EXPECT_LE(actualGenome._mutationRates._cellTypeModeMutation._nodeProbability, 1.0f);
}

TEST_F(MetaMutationTests, metaMutation_cellTypeModeRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypeModeMutation = CellTypeModeMutationDesc().nodeProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.cellTypeModeMetaMutationsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._cellTypeModeMutation._nodeProbability, 0.5f);
}

TEST_F(MetaMutationTests, metaMutation_cellTypeRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().nodeProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.cellTypeMetaMutationsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_NE(actualGenome._mutationRates._cellTypeMutation._nodeProbability, 0.5f);
    EXPECT_GE(actualGenome._mutationRates._cellTypeMutation._nodeProbability, 0.0f);
    EXPECT_LE(actualGenome._mutationRates._cellTypeMutation._nodeProbability, 1.0f);
}

TEST_F(MetaMutationTests, metaMutation_cellTypeRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().nodeProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.cellTypeMetaMutationsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._cellTypeMutation._nodeProbability, 0.5f);
}

TEST_F(MetaMutationTests, metaMutation_voidRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._voidMutation = VoidMutationDesc().nodeProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.voidMetaMutationsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_FALSE(approxCompare(actualGenome._mutationRates._voidMutation._nodeProbability, 0.5f));
}

TEST_F(MetaMutationTests, metaMutation_voidRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._voidMutation = VoidMutationDesc().nodeProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.voidMetaMutationsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._voidMutation._nodeProbability, 0.5f);
}

TEST_F(MetaMutationTests, metaMutation_appendNodeRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._appendNodeMutation = AppendNodeMutationDesc().geneProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.appendNodeMetaMutationsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_FALSE(approxCompare(actualGenome._mutationRates._appendNodeMutation._geneProbability, 0.5f));
    EXPECT_GE(actualGenome._mutationRates._appendNodeMutation._geneProbability, 0.0f);
    EXPECT_LE(actualGenome._mutationRates._appendNodeMutation._geneProbability, 1.0f);
}

TEST_F(MetaMutationTests, metaMutation_appendNodeRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._appendNodeMutation = AppendNodeMutationDesc().geneProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.appendNodeMetaMutationsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._appendNodeMutation._geneProbability, 0.5f);
}

TEST_F(MetaMutationTests, metaMutation_addNodeRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._addNodeMutation = AddNodeMutationDesc().geneProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.addNodeMetaMutationsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_FALSE(approxCompare(actualGenome._mutationRates._addNodeMutation._geneProbability, 0.5f));
    EXPECT_GE(actualGenome._mutationRates._addNodeMutation._geneProbability, 0.0f);
    EXPECT_LE(actualGenome._mutationRates._addNodeMutation._geneProbability, 1.0f);
}

TEST_F(MetaMutationTests, metaMutation_addNodeRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._addNodeMutation = AddNodeMutationDesc().geneProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.addNodeMetaMutationsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._addNodeMutation._geneProbability, 0.5f);
}

TEST_F(MetaMutationTests, metaMutation_trimNodeRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._trimNodeMutation = TrimNodeMutationDesc().geneProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.trimNodeMetaMutationsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_FALSE(approxCompare(actualGenome._mutationRates._trimNodeMutation._geneProbability, 0.5f));
    EXPECT_GE(actualGenome._mutationRates._trimNodeMutation._geneProbability, 0.0f);
    EXPECT_LE(actualGenome._mutationRates._trimNodeMutation._geneProbability, 1.0f);
}

TEST_F(MetaMutationTests, metaMutation_trimNodeRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._trimNodeMutation = TrimNodeMutationDesc().geneProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.trimNodeMetaMutationsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._trimNodeMutation._geneProbability, 0.5f);
}

TEST_F(MetaMutationTests, metaMutation_deleteNodeRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._deleteNodeMutation = DeleteNodeMutationDesc().geneProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.deleteNodeMetaMutationsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_FALSE(approxCompare(actualGenome._mutationRates._deleteNodeMutation._geneProbability, 0.5f));
    EXPECT_GE(actualGenome._mutationRates._deleteNodeMutation._geneProbability, 0.0f);
    EXPECT_LE(actualGenome._mutationRates._deleteNodeMutation._geneProbability, 1.0f);
}

TEST_F(MetaMutationTests, metaMutation_deleteNodeRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._deleteNodeMutation = DeleteNodeMutationDesc().geneProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.deleteNodeMetaMutationsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._deleteNodeMutation._geneProbability, 0.5f);
}

TEST_F(MetaMutationTests, metaMutation_constructorRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._constructorMutations[0] =
        ConstructorMutationDesc().nodeProbability(0.5f).valueChangeSigma(0.5f).enumChangeProbability(0.5f).constructorToggleProbability(0.5f);
    genome._mutationRates._constructorMutations[1] =
        ConstructorMutationDesc().nodeProbability(0.4f).valueChangeSigma(0.4f).enumChangeProbability(0.4f).constructorToggleProbability(0.4f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.constructorMetaMutationsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    bool anyChanged = actualGenome._mutationRates._constructorMutations[0]._nodeProbability != 0.5f
        || actualGenome._mutationRates._constructorMutations[0]._valueChangeSigma != 0.5f
        || actualGenome._mutationRates._constructorMutations[0]._enumChangeProbability != 0.5f
        || actualGenome._mutationRates._constructorMutations[0]._constructorToggleProbability != 0.5f
        || actualGenome._mutationRates._constructorMutations[1]._nodeProbability != 0.4f
        || actualGenome._mutationRates._constructorMutations[1]._valueChangeSigma != 0.4f
        || actualGenome._mutationRates._constructorMutations[1]._enumChangeProbability != 0.4f
        || actualGenome._mutationRates._constructorMutations[1]._constructorToggleProbability != 0.4f;
    EXPECT_TRUE(anyChanged);
    EXPECT_GE(actualGenome._mutationRates._constructorMutations[0]._nodeProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._constructorMutations[0]._valueChangeSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._constructorMutations[0]._enumChangeProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._constructorMutations[0]._constructorToggleProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._constructorMutations[1]._nodeProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._constructorMutations[1]._valueChangeSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._constructorMutations[1]._enumChangeProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._constructorMutations[1]._constructorToggleProbability, 0.0f);
}

TEST_F(MetaMutationTests, metaMutation_constructorRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._constructorMutations[0] =
        ConstructorMutationDesc().nodeProbability(0.5f).valueChangeSigma(0.5f).enumChangeProbability(0.5f).constructorToggleProbability(0.5f);
    genome._mutationRates._constructorMutations[1] =
        ConstructorMutationDesc().nodeProbability(0.4f).valueChangeSigma(0.4f).enumChangeProbability(0.4f).constructorToggleProbability(0.4f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.constructorMetaMutationsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._constructorMutations[0]._nodeProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._constructorMutations[0]._valueChangeSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._constructorMutations[0]._enumChangeProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._constructorMutations[0]._constructorToggleProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._constructorMutations[1]._nodeProbability, 0.4f);
    EXPECT_EQ(actualGenome._mutationRates._constructorMutations[1]._valueChangeSigma, 0.4f);
    EXPECT_EQ(actualGenome._mutationRates._constructorMutations[1]._enumChangeProbability, 0.4f);
    EXPECT_EQ(actualGenome._mutationRates._constructorMutations[1]._constructorToggleProbability, 0.4f);
}

TEST_F(MetaMutationTests, metaMutation_applyMetaMutationsDisabledNoRateChange)
{
    auto genome = createTestGenome();
    genome._applyMetaMutations = false;
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().nodeProbability(0.5f).weightChangeSigma(0.5f).biasChangeSigma(0.5f).actfnChangeProbability(0.5f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().nodeProbability(0.5f).weightChangeSigma(0.5f).biasChangeSigma(0.5f).actfnChangeProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.neuronsMetaMutationsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._nodeProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._weightChangeSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._biasChangeSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._actfnChangeProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._nodeProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._weightChangeSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._biasChangeSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._actfnChangeProbability, 0.5f);
}
