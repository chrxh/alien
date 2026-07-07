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

TEST_F(ConstructorMutationTests, mutatesCreatureWhileConstructingOffspring)
{
    // Regression test:
    // constructor.offspring is set on the first energy-less trigger and, with separation off, never reset.
    // External energy inflow then lifts the energy until the offspring is actually constructed and the creature
    // is mutated - which the previous code skipped while constructor.offspring != nullptr.
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc()})});
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().nodeProbability(1.0f).weightChangeSigma(1.0f);

    auto data = Desc().addCreature(
        {ObjectDesc()
             .id(1)
             .pos({100.0f, 100.0f})
             .type(CellDesc().usableEnergy(0.0f).constructor(ConstructorDesc().autoTriggerInterval(1).geneIndex(0).separation(false)))},
        CreatureDesc().id(1).mutationState(MutationState_NotMutated),
        genome);

    _parameters.externalEnergyControlToggle.value = true;
    _parameters.externalEnergy.value = 1000.0f;
    _parameters.newLineageThreshold.value = 100.0f;  // keep accumulatedMutations from resetting
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 50; ++i) {
        _simulationFacade->testOnly_calcTimestepWithCellFunctions();
    }

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.getNumObjects());  // offspring cell was constructed
    auto hostCreatureId = actualData.getObjectRef(1).getCellRef()._creatureId;
    EXPECT_GT(actualData.getCreatureRef(hostCreatureId)._accumulatedMutations, 0.0f);

    auto offspringCreatureId = actualData.getOtherObjectRef(1).getCellRef()._creatureId;
    EXPECT_GT(actualData.getCreatureRef(offspringCreatureId)._accumulatedMutations, 0.0f);
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
