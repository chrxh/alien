#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class AccumulatedMutationTests : public MutationTestsBase
{
};

enum class MutationType
{
    Neuron,
    Connection,
    CellTypeProperties,
    Geometry,
    CellTypeMode,
    CellType,
    Void,
    ExtendGene,
    AddNode,
    TrimGene,
    DeleteNode,
    Constructor
};

class AccumulatedMutationTests_AllTypes
    : public AccumulatedMutationTests
    , public ::testing::WithParamInterface<MutationType>
{
};

INSTANTIATE_TEST_SUITE_P(
    AccumulatedMutationTests,
    AccumulatedMutationTests_AllTypes,
    ::testing::Values(
        MutationType::Neuron,
        MutationType::Connection,
        MutationType::CellTypeProperties,
        MutationType::Geometry,
        MutationType::CellTypeMode,
        MutationType::CellType,
        MutationType::Void,
        MutationType::ExtendGene,
        MutationType::AddNode,
        MutationType::TrimGene,
        MutationType::DeleteNode,
        MutationType::Constructor));

TEST_P(AccumulatedMutationTests_AllTypes, accumulatedMutations_increases)
{
    auto genome = createTestGenome();
    switch (GetParam()) {
    case MutationType::Neuron:
        genome._mutationRates._neuronMutations[0] =
            NeuronMutationDesc().nodeProbability(1.0f).weightChangeSigma(1.0f).biasChangeSigma(1.0f).actfnChangeProbability(1.0f);
        break;
    case MutationType::Connection:
        genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f);
        break;
    case MutationType::CellTypeProperties:
        genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);
        break;
    case MutationType::Geometry:
        genome._mutationRates._geometryMutations[0] = GeometryMutationDesc().geneProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);
        break;
    case MutationType::CellTypeMode:
        genome._mutationRates._cellTypeModeMutation = CellTypeModeMutationDesc().nodeProbability(1.0f);
        break;
    case MutationType::CellType:
        genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().nodeProbability(1.0f);
        break;
    case MutationType::Void:
        genome._mutationRates._voidMutation = VoidMutationDesc().nodeProbability(1.0f);
        break;
    case MutationType::ExtendGene:
        genome._mutationRates._extendGeneMutation = ExtendGeneMutationDesc().geneProbability(1.0f);
        break;
    case MutationType::AddNode:
        // Per-node probability: inserting before every node compounds the genome each pass, so a full probability would
        // exhaust the genome heap over the 100 passes below. A small probability still accumulates mutations reliably.
        genome._mutationRates._addNodeMutation = AddNodeMutationDesc().nodeProbability(0.02f);
        break;
    case MutationType::TrimGene:
        genome._mutationRates._trimGeneMutation = TrimGeneMutationDesc().geneProbability(1.0f);
        break;
    case MutationType::DeleteNode:
        genome._mutationRates._deleteNodeMutation = DeleteNodeMutationDesc().nodeProbability(1.0f);
        break;
    case MutationType::Constructor:
        genome._mutationRates._constructorMutations[0] =
            ConstructorMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f).constructorToggleProbability(1.0f);
        break;
    }

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc().lineageId(42), genome);

    auto parameters = _parameters;
    parameters.newLineageThreshold.value = 1.0e30f;
    _simulationFacade->setSimulationParameters(parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualCreature = getMutatedCreature();
    EXPECT_GT(actualCreature._accumulatedMutations, 0.0f);
    EXPECT_GT(actualCreature._accumulatedMutationsInLineage, 0.0f);
}

TEST_F(AccumulatedMutationTests, accumulatedMutations_metaMutationDoesNotAccount)
{
    auto genome = GenomeDesc();

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc().lineageId(42), genome);

    _parameters.neuronsMetaMutationsSigma.value = 1.0f;
    _parameters.connectionsMetaMutationsSigma.value = 1.0f;
    _parameters.cellTypePropertiesMetaMutationsSigma.value = 1.0f;
    _parameters.geometryMetaMutationsSigma.value = 1.0f;
    _parameters.cellTypeModeMetaMutationsSigma.value = 1.0f;
    _parameters.cellTypeMetaMutationsSigma.value = 1.0f;
    _parameters.voidMetaMutationsSigma.value = 1.0f;
    _parameters.extendGeneMetaMutationsSigma.value = 1.0f;
    _parameters.addNodeMetaMutationsSigma.value = 1.0f;
    _parameters.trimGeneMetaMutationsSigma.value = 1.0f;
    _parameters.deleteNodeMetaMutationsSigma.value = 1.0f;
    _parameters.constructorMetaMutationsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualCreature = getMutatedCreature();
    EXPECT_EQ(actualCreature._accumulatedMutations, 0.0f);
    EXPECT_EQ(actualCreature._accumulatedMutationsInLineage, 0.0f);
    EXPECT_EQ(actualCreature._lineageId, 42);
}

TEST_F(AccumulatedMutationTests, accumulatedMutations_createsNewLineageId)
{
    auto genome = createTestGenome();

    auto data =
        Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc().lineageId(42).accumulatedMutations(11.0f).accumulatedMutationsInLineage(11.0f), genome);

    _parameters.newLineageThreshold.value = 0.1f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualCreature = getMutatedCreature();
    EXPECT_GT(actualCreature._lineageId, 42);
    EXPECT_EQ(actualCreature._accumulatedMutationsInLineage, 0.0f);
    EXPECT_GT(actualCreature._accumulatedMutations, 11.0f);
}
