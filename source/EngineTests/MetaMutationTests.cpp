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
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().eventProbability(0.5f).weightSigma(0.5f).biasSigma(0.5f).activationFunctionProbability(0.5f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().eventProbability(0.5f).weightSigma(0.5f).biasSigma(0.5f).activationFunctionProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationNeuronsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    bool anyChanged = actualGenome._mutationRates._neuronMutations[0]._eventProbability != 0.5f
        || actualGenome._mutationRates._neuronMutations[0]._weightSigma != 0.5f || actualGenome._mutationRates._neuronMutations[0]._biasSigma != 0.5f
        || actualGenome._mutationRates._neuronMutations[0]._activationFunctionProbability != 0.5f
        || actualGenome._mutationRates._neuronMutations[1]._eventProbability != 0.5f || actualGenome._mutationRates._neuronMutations[1]._weightSigma != 0.5f
        || actualGenome._mutationRates._neuronMutations[1]._biasSigma != 0.5f
        || actualGenome._mutationRates._neuronMutations[1]._activationFunctionProbability != 0.5f;
    EXPECT_TRUE(anyChanged);

    EXPECT_GE(actualGenome._mutationRates._neuronMutations[0]._eventProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[0]._weightSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[0]._biasSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[0]._activationFunctionProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[1]._eventProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[1]._weightSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[1]._biasSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[1]._activationFunctionProbability, 0.0f);
}

TEST_F(MetaMutationTests, metaMutation_neuronRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().eventProbability(0.5f).weightSigma(0.5f).biasSigma(0.5f).activationFunctionProbability(0.5f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().eventProbability(0.5f).weightSigma(0.5f).biasSigma(0.5f).activationFunctionProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationNeuronsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._eventProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._weightSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._biasSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._activationFunctionProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._eventProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._weightSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._biasSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._activationFunctionProbability, 0.5f);
}

TEST_F(MetaMutationTests, metaMutation_connectionRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().eventProbability(0.5f).sigma(0.5f);
    genome._mutationRates._connectionMutations[1] = ConnectionMutationDesc().eventProbability(0.5f).sigma(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationConnectionsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    bool anyChanged = actualGenome._mutationRates._connectionMutations[0]._eventProbability != 0.5f
        || actualGenome._mutationRates._connectionMutations[0]._sigma != 0.5f || actualGenome._mutationRates._connectionMutations[1]._eventProbability != 0.5f
        || actualGenome._mutationRates._connectionMutations[1]._sigma != 0.5f;
    EXPECT_TRUE(anyChanged);
    EXPECT_GE(actualGenome._mutationRates._connectionMutations[0]._eventProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._connectionMutations[0]._sigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._connectionMutations[1]._eventProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._connectionMutations[1]._sigma, 0.0f);
}

TEST_F(MetaMutationTests, metaMutation_connectionRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().eventProbability(0.5f).sigma(0.5f);
    genome._mutationRates._connectionMutations[1] = ConnectionMutationDesc().eventProbability(0.5f).sigma(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationConnectionsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_EQ(actualGenome._mutationRates._connectionMutations[0]._eventProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._connectionMutations[0]._sigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._connectionMutations[1]._eventProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._connectionMutations[1]._sigma, 0.5f);
}

TEST_F(MetaMutationTests, metaMutation_cellTypePropertyRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().eventProbability(0.5f).sigma(0.5f).probability(0.5f);
    genome._mutationRates._cellTypePropertiesMutations[1] = CellTypePropertiesMutationDesc().eventProbability(0.4f).sigma(0.4f).probability(0.4f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationCellTypePropertiesSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    bool anyChanged = actualGenome._mutationRates._cellTypePropertiesMutations[0]._eventProbability != 0.5f
        || actualGenome._mutationRates._cellTypePropertiesMutations[0]._sigma != 0.5f
        || actualGenome._mutationRates._cellTypePropertiesMutations[0]._probability != 0.5f
        || actualGenome._mutationRates._cellTypePropertiesMutations[1]._eventProbability != 0.4f
        || actualGenome._mutationRates._cellTypePropertiesMutations[1]._sigma != 0.4f
        || actualGenome._mutationRates._cellTypePropertiesMutations[1]._probability != 0.4f;
    EXPECT_TRUE(anyChanged);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[0]._eventProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[0]._sigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[0]._probability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[1]._eventProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[1]._sigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[1]._probability, 0.0f);
}

TEST_F(MetaMutationTests, metaMutation_cellTypePropertyRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().eventProbability(0.5f).sigma(0.5f).probability(0.5f);
    genome._mutationRates._cellTypePropertiesMutations[1] = CellTypePropertiesMutationDesc().eventProbability(0.4f).sigma(0.4f).probability(0.4f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationCellTypePropertiesSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[0]._eventProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[0]._sigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[0]._probability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[1]._eventProbability, 0.4f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[1]._sigma, 0.4f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[1]._probability, 0.4f);
}

TEST_F(MetaMutationTests, metaMutation_cellTypeModeRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypeModeMutation = CellTypeModeMutationDesc().eventProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationCellTypeModeSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_NE(actualGenome._mutationRates._cellTypeModeMutation._eventProbability, 0.5f);
    EXPECT_GE(actualGenome._mutationRates._cellTypeModeMutation._eventProbability, 0.0f);
    EXPECT_LE(actualGenome._mutationRates._cellTypeModeMutation._eventProbability, 1.0f);
}

TEST_F(MetaMutationTests, metaMutation_cellTypeModeRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypeModeMutation = CellTypeModeMutationDesc().eventProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationCellTypeModeSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._cellTypeModeMutation._eventProbability, 0.5f);
}

TEST_F(MetaMutationTests, metaMutation_cellTypeRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().eventProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationCellTypeSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_NE(actualGenome._mutationRates._cellTypeMutation._eventProbability, 0.5f);
    EXPECT_GE(actualGenome._mutationRates._cellTypeMutation._eventProbability, 0.0f);
    EXPECT_LE(actualGenome._mutationRates._cellTypeMutation._eventProbability, 1.0f);
}

TEST_F(MetaMutationTests, metaMutation_cellTypeRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().eventProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationCellTypeSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._cellTypeMutation._eventProbability, 0.5f);
}

TEST_F(MetaMutationTests, metaMutation_voidRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._voidMutation = VoidMutationDesc().eventProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationVoidSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_FALSE(approxCompare(actualGenome._mutationRates._voidMutation._eventProbability, 0.5f));
}

TEST_F(MetaMutationTests, metaMutation_voidRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._voidMutation = VoidMutationDesc().eventProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationVoidSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._voidMutation._eventProbability, 0.5f);
}
