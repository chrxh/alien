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
    CellTypeMode,
    CellType,
    Void,
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
        MutationType::CellTypeMode,
        MutationType::CellType,
        MutationType::Void,
        MutationType::Constructor));

TEST_P(AccumulatedMutationTests_AllTypes, accumulatedMutations_increases)
{
    auto genome = createTestGenome().lineageId(42).prevLineageId(41);
    switch (GetParam()) {
    case MutationType::Neuron:
        genome._mutationRates._neuronMutations[0] =
            NeuronMutationDesc().nodeProbability(1.0f).weightSigma(1.0f).biasSigma(1.0f).activationFunctionProbability(1.0f);
        break;
    case MutationType::Connection:
        genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().nodeProbability(1.0f).sigma(1.0f);
        break;
    case MutationType::CellTypeProperties:
        genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().nodeProbability(1.0f).sigma(1.0f).discreteChangeProbability(1.0f);
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
    case MutationType::Constructor:
        genome._mutationRates._constructorMutations[0] =
            ConstructorMutationDesc().nodeProbability(1.0f).sigma(1.0f).discreteChangeProbability(1.0f).existConstructorProbability(1.0f);
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

    _parameters.metaMutationNeuronsSigma.value = 1.0f;
    _parameters.metaMutationConnectionsSigma.value = 1.0f;
    _parameters.metaMutationCellTypePropertiesSigma.value = 1.0f;
    _parameters.metaMutationCellTypeModeSigma.value = 1.0f;
    _parameters.metaMutationCellTypeSigma.value = 1.0f;
    _parameters.metaMutationVoidSigma.value = 1.0f;
    _parameters.metaMutationConstructorSigma.value = 1.0f;
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
