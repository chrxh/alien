#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class AccumulatedMutationTests : public MutationTestsBase
{
};

enum class MutationCategory
{
    Neuron,
    Connection,
    CellTypeProperties,
    CellTypeMode,
    CellType,
    Void,
    AppendNode,
    AddNode,
    TrimNode,
    DeleteNode,
    Constructor
};

class AccumulatedMutationTests_AllTypes
    : public AccumulatedMutationTests
    , public ::testing::WithParamInterface<MutationCategory>
{
};

INSTANTIATE_TEST_SUITE_P(
    AccumulatedMutationTests,
    AccumulatedMutationTests_AllTypes,
    ::testing::Values(
        MutationCategory::Neuron,
        MutationCategory::Connection,
        MutationCategory::CellTypeProperties,
        MutationCategory::CellTypeMode,
        MutationCategory::CellType,
        MutationCategory::Void,
        MutationCategory::AppendNode,
        MutationCategory::AddNode,
        MutationCategory::TrimNode,
        MutationCategory::DeleteNode,
        MutationCategory::Constructor));

TEST_P(AccumulatedMutationTests_AllTypes, accumulatedMutations_increases)
{
    auto genome = createTestGenome().lineageId(42).prevLineageId(41);
    switch (GetParam()) {
    case MutationCategory::Neuron:
        genome._mutationRates._neuronMutations[0] =
            NeuronMutationDesc().nodeProbability(1.0f).weightChangeSigma(1.0f).biasChangeSigma(1.0f).actfnChangeProbability(1.0f);
        break;
    case MutationCategory::Connection:
        genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f);
        break;
    case MutationCategory::CellTypeProperties:
        genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);
        break;
    case MutationCategory::CellTypeMode:
        genome._mutationRates._cellTypeModeMutation = CellTypeModeMutationDesc().nodeProbability(1.0f);
        break;
    case MutationCategory::CellType:
        genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().nodeProbability(1.0f);
        break;
    case MutationCategory::Void:
        genome._mutationRates._voidMutation = VoidMutationDesc().nodeProbability(1.0f);
        break;
    case MutationCategory::AppendNode:
        genome._mutationRates._appendNodeMutation = AppendNodeMutationDesc().geneProbability(1.0f);
        break;
    case MutationCategory::AddNode:
        genome._mutationRates._addNodeMutation = AddNodeMutationDesc().geneProbability(1.0f);
        break;
    case MutationCategory::TrimNode:
        genome._mutationRates._trimNodeMutation = TrimNodeMutationDesc().geneProbability(1.0f);
        break;
    case MutationCategory::DeleteNode:
        genome._mutationRates._deleteNodeMutation = DeleteNodeMutationDesc().geneProbability(1.0f);
        break;
    case MutationCategory::Constructor:
        genome._mutationRates._constructorMutations[0] =
            ConstructorMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f).constructorToggleProbability(1.0f);
        break;
    }

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    auto parameters = _parameters;
    parameters.newLineageThreshold.value = 1.0e30f;
    _simulationFacade->setSimulationParameters(parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_GT(actualGenome._accumulatedMutations, genome._accumulatedMutations);
}

TEST_F(AccumulatedMutationTests, accumulatedMutations_metaMutationDoesNotAccount)
{
    auto genome = GenomeDesc().lineageId(42).prevLineageId(41);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.neuronsMetaMutationsSigma.value = 1.0f;
    _parameters.connectionsMetaMutationsSigma.value = 1.0f;
    _parameters.cellTypePropertiesMetaMutationsSigma.value = 1.0f;
    _parameters.cellTypeModeMetaMutationsSigma.value = 1.0f;
    _parameters.cellTypeMetaMutationsSigma.value = 1.0f;
    _parameters.voidMetaMutationsSigma.value = 1.0f;
    _parameters.appendNodeMetaMutationsSigma.value = 1.0f;
    _parameters.addNodeMetaMutationsSigma.value = 1.0f;
    _parameters.trimNodeMetaMutationsSigma.value = 1.0f;
    _parameters.deleteNodeMetaMutationsSigma.value = 1.0f;
    _parameters.constructorMetaMutationsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._accumulatedMutations, 0.0f);
    EXPECT_EQ(actualGenome._lineageId, 42);
    EXPECT_EQ(actualGenome._prevLineageId, 41);
}

TEST_F(AccumulatedMutationTests, accumulatedMutations_createsNewLineageId)
{
    auto genome = createTestGenome().lineageId(42).prevLineageId(41);
    genome._accumulatedMutations = 11.0f;

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.newLineageThreshold.value = 0.1f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_TRUE(actualGenome._prevLineageId.has_value());
    EXPECT_EQ(*actualGenome._prevLineageId, 42);
    EXPECT_GT(actualGenome._lineageId, 42);
}
