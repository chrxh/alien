#include <optional>

#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class ConstructorMutationTests : public MutationTestsBase
{
protected:
    // Resets the optional constructor of every node so that two genomes can be compared ignoring constructor attributes.
    bool compareAllExceptConstructor(GenomeDesc expected, GenomeDesc actual)
    {
        auto reset = [](GenomeDesc& genome) {
            for (auto& gene : genome._genes) {
                for (auto& node : gene._nodes) {
                    node._constructor.reset();
                }
            }
        };
        reset(expected);
        reset(actual);
        return expected == actual;
    }
};

TEST_F(ConstructorMutationTests, constructorMutation_changesConstructorAttributes)
{
    auto genome = createTestGenome();
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    auto const& original = genome._genes.at(0)._nodes.at(0)._constructor.value();

    _simulationFacade->setSimulationData(data);

    // Every mutable constructor attribute must change at least once (provideEnergy is intentionally not mutated).
    _simulationFacade->testOnly_mutate(1);
    auto const actualGenome = getMutatedGenome();
    auto const& constructor = actualGenome._genes.at(0)._nodes.at(0)._constructor;
    ASSERT_TRUE(constructor.has_value());  // without constructorToggleProbability the constructor must never be removed
    auto const& mutated = constructor.value();

    EXPECT_TRUE(mutated._autoTriggerInterval != original._autoTriggerInterval);
    EXPECT_TRUE(mutated._geneIndex != original._geneIndex);
    EXPECT_TRUE(mutated._constructionActivationTime != original._constructionActivationTime);
    EXPECT_TRUE(mutated._constructionAngle != original._constructionAngle);
    EXPECT_TRUE(mutated._reservedEnergy != original._reservedEnergy);
    EXPECT_TRUE(mutated._separation != original._separation);
    EXPECT_TRUE(mutated._numBranches != original._numBranches);
    EXPECT_TRUE(mutated._numConcatenations != original._numConcatenations);
}

TEST_F(ConstructorMutationTests, constructorMutation_addsConstructorWithDefaultValues)
{
    auto genome = createTestGenome();
    genome._genes.at(0)._nodes.at(0)._constructor.reset();  // node without a constructor
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().nodeProbability(1.0f).constructorToggleProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();

    // The node had no constructor, so turning it on must initialize it with default values.
    auto const& constructor = actualGenome._genes.at(0)._nodes.at(0)._constructor;
    ASSERT_TRUE(constructor.has_value());
    EXPECT_TRUE(constructor.value() == ConstructorGenomeDesc());
}

TEST_F(ConstructorMutationTests, mutatesCreatureWhileOffspringUnderConstruction)
{
    // The mutation gate in MutationProcessor::process no longer requires constructor.offspring == nullptr,
    // so a creature is mutated even while its constructor still has an offspring under construction.
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().nodeProbability(1.0f).weightChangeSigma(1.0f);

    auto data = Desc().addCreature(
        {ObjectDesc()
             .id(1)
             .pos({100.0f, 100.0f})
             .type(CellDesc()
                       .usableEnergy(_parameters.normalCellEnergy.value[0] * 3.5f)  // enough energy for mutation and construction
                       .constructor(ConstructorDesc().geneIndex(0).separation(false).numBranches(1).numConcatenations(1)))},
        CreatureDesc().id(1).mutationState(MutationState_NotMutated),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    // The offspring construction has started but is not finished (the gene has two nodes, one built so far).
    ASSERT_EQ(2, actualData._creatures.size());
    auto newCreature = actualData.getOtherCreatureRef(1);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());

    // Despite the offspring still being under construction, the host genome must have been mutated.
    auto mutatedGenome = getMutatedGenome(1);
    bool anyWeightChanged = false;
    for (size_t g = 0; g < genome._genes.size() && !anyWeightChanged; ++g) {
        for (size_t n = 0; n < genome._genes.at(g)._nodes.size() && !anyWeightChanged; ++n) {
            auto const& origWeights = genome._genes.at(g)._nodes.at(n)._neuralNetwork._weights;
            auto const& actualWeights = mutatedGenome._genes.at(g)._nodes.at(n)._neuralNetwork._weights;
            for (size_t w = 0; w < origWeights.size(); ++w) {
                if (origWeights.at(w) != actualWeights.at(w)) {
                    anyWeightChanged = true;
                    break;
                }
            }
        }
    }
    EXPECT_TRUE(anyWeightChanged);
}

TEST_F(ConstructorMutationTests, constructorMutation_zeroProbabilityNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._constructorMutations[0] =
        ConstructorMutationDesc().nodeProbability(0.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f).constructorToggleProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome, genome);
}

TEST_F(ConstructorMutationTests, constructorMutation_keepOtherAttributesUnchanged)
{
    auto genome = createTestGenome();
    genome._mutationRates._constructorMutations[0] =
        ConstructorMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f).constructorToggleProbability(1.0f);
    genome._mutationRates._constructorMutations[1] =
        ConstructorMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f).constructorToggleProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    EXPECT_TRUE(compareAllExceptConstructor(genome, actualGenome));
}

TEST_F(ConstructorMutationTests, constructorMutation_existProbabilityTogglesConstructorPresence)
{
    auto genome = createTestGenome();
    genome._genes.at(0)._nodes.at(0)._constructor.reset();  // one node without a constructor
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().nodeProbability(1.0f).constructorToggleProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);  // a single step flips the presence of every node exactly once

    auto actualGenome = getMutatedGenome();

    // constructorToggleProbability alone must toggle whether a node has a constructor.
    EXPECT_TRUE(actualGenome._genes.at(0)._nodes.at(0)._constructor.has_value());   // had none -> gained one
    EXPECT_FALSE(actualGenome._genes.at(0)._nodes.at(1)._constructor.has_value());  // had one -> lost it
}
