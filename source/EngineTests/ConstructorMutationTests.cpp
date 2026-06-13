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
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().eventProbability(1.0f).sigma(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    auto const numGenes = static_cast<int>(actualGenome._genes.size());

    auto const& originalConstructor = genome._genes.at(0)._nodes.at(0)._constructor;
    auto const& actualConstructor = actualGenome._genes.at(0)._nodes.at(0)._constructor;
    ASSERT_TRUE(originalConstructor.has_value());
    ASSERT_TRUE(actualConstructor.has_value());
    EXPECT_NE(originalConstructor->_constructionAngle, actualConstructor->_constructionAngle);

    // All mutated constructors must stay within their valid ranges.
    for (auto const& gene : actualGenome._genes) {
        for (auto const& node : gene._nodes) {
            if (node._constructor.has_value()) {
                auto const& constructor = node._constructor.value();
                EXPECT_GE(constructor._geneIndex, 0);
                EXPECT_LT(constructor._geneIndex, numGenes);
                EXPECT_GE(constructor._numBranches, 1);
                EXPECT_LE(constructor._numBranches, 6);
                EXPECT_GE(constructor._reservedEnergy, 0.0f);
                EXPECT_GE(constructor._numConcatenations, 1);
                EXPECT_GE(constructor._constructionAngle, Const::ConstructorConstructionAngle_Min);
                EXPECT_LE(constructor._constructionAngle, Const::ConstructorConstructionAngle_Max);
            }
        }
    }
}

TEST_F(ConstructorMutationTests, constructorMutation_addsConstructorWithDefaultValues)
{
    auto genome = createTestGenome();
    genome._genes.at(0)._nodes.at(0)._constructor.reset();  // node without a constructor
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().eventProbability(1.0f).probability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();

    // The node had no constructor, so turning it on must initialize it with default values.
    auto const& constructor = actualGenome._genes.at(0)._nodes.at(0)._constructor;
    ASSERT_TRUE(constructor.has_value());
    EXPECT_TRUE(constructor.value() == ConstructorGenomeDesc());
}

TEST_F(ConstructorMutationTests, constructorMutation_doesNotChangeOtherAttributes)
{
    auto genome = createTestGenome();
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().eventProbability(1.0f).sigma(1.0f).probability(1.0f);
    genome._mutationRates._constructorMutations[1] = ConstructorMutationDesc().eventProbability(1.0f).sigma(1.0f).probability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    EXPECT_TRUE(compareAllExceptConstructor(genome, actualGenome));
}

TEST_F(ConstructorMutationTests, constructorMutation_zeroProbabilityNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().eventProbability(0.0f).sigma(1.0f).probability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome, genome);
}
