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
            genome._lineageId = 0;
            genome._prevLineageId = std::nullopt;
            genome._accumulatedMutations = 0.0f;
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
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().nodeProbability(1.0f).sigma(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    auto const numGenes = static_cast<int>(actualGenome._genes.size());

    auto const& originalConstructor = genome._genes.at(0)._nodes.at(0)._constructor;
    auto const& actualConstructor = actualGenome._genes.at(0)._nodes.at(0)._constructor;
    ASSERT_TRUE(originalConstructor.has_value());
    ASSERT_TRUE(actualConstructor.has_value());
    EXPECT_NE(originalConstructor->_constructionAngle, actualConstructor->_constructionAngle);
    EXPECT_NE(originalConstructor->_reservedEnergy, actualConstructor->_reservedEnergy);

    // All mutated constructors must stay within their valid ranges.
    if (originalConstructor->_autoTriggerInterval.has_value()) {
        EXPECT_GE(actualConstructor->_autoTriggerInterval.value(), Const::ConstructorAutoTriggerInterval_Min);
    }
    EXPECT_GE(originalConstructor->_geneIndex, 0);
    EXPECT_LT(originalConstructor->_geneIndex, numGenes);
    EXPECT_GE(originalConstructor->_numBranches, 1);
    EXPECT_LE(originalConstructor->_numBranches, 6);
    EXPECT_GE(originalConstructor->_reservedEnergy, 0.0f);
    EXPECT_GE(originalConstructor->_numConcatenations, 1);
    EXPECT_GE(originalConstructor->_constructionAngle, Const::ConstructorConstructionAngle_Min);
    EXPECT_LE(originalConstructor->_constructionAngle, Const::ConstructorConstructionAngle_Max);
    EXPECT_GE(originalConstructor->_constructionActivationTime, Const::ConstructorConstructionActivationTime_Min);
    EXPECT_LE(originalConstructor->_constructionActivationTime, Const::ConstructorConstructionActivationTime_Max);
}

TEST_F(ConstructorMutationTests, constructorMutation_mutatesManualAutoTriggerInterval)
{
    auto genome = GenomeDesc().genes(
        {GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().autoTriggerInterval(std::nullopt).geneIndex(0).separation(false))})});
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().nodeProbability(1.0f).sigma(0.0f).discreteChangeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    auto const& actualConstructor = actualGenome._genes.at(0)._nodes.at(0)._constructor;
    ASSERT_TRUE(actualConstructor.has_value());
    EXPECT_EQ(actualConstructor->_autoTriggerInterval, Const::ConstructorAutoTriggerInterval_Default);
}

TEST_F(ConstructorMutationTests, constructorMutation_addsConstructorWithDefaultValues)
{
    auto genome = createTestGenome();
    genome._genes.at(0)._nodes.at(0)._constructor.reset();  // node without a constructor
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().nodeProbability(1.0f).discreteChangeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();

    // The node had no constructor, so turning it on must initialize it with default values.
    auto const& constructor = actualGenome._genes.at(0)._nodes.at(0)._constructor;
    ASSERT_TRUE(constructor.has_value());
    EXPECT_TRUE(constructor.value() == ConstructorGenomeDesc());
}

TEST_F(ConstructorMutationTests, constructorMutation_zeroProbabilityNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().nodeProbability(0.0f).sigma(1.0f).discreteChangeProbability(1.0f);

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
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().nodeProbability(1.0f).sigma(1.0f).discreteChangeProbability(1.0f);
    genome._mutationRates._constructorMutations[1] = ConstructorMutationDesc().nodeProbability(1.0f).sigma(1.0f).discreteChangeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    EXPECT_TRUE(compareAllExceptConstructor(genome, actualGenome));
}
