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

TEST_F(ConstructorMutationTests, constructorMutation_changesScalarProperties)
{
    auto genome = createTestGenome();
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().eventProbability(1.0f).sigma(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();

    auto const& originalConstructor = genome._genes.at(0)._nodes.at(0)._constructor;
    auto const& actualConstructor = actualGenome._genes.at(0)._nodes.at(0)._constructor;
    ASSERT_TRUE(originalConstructor.has_value());
    ASSERT_TRUE(actualConstructor.has_value());
    EXPECT_NE(originalConstructor->_constructionAngle, actualConstructor->_constructionAngle);
}

TEST_F(ConstructorMutationTests, constructorMutation_changesBoolAndEnumProperties)
{
    auto genome = createTestGenome();
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().eventProbability(1.0f).probability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();

    auto const& originalConstructor = genome._genes.at(0)._nodes.at(0)._constructor;
    auto const& actualConstructor = actualGenome._genes.at(0)._nodes.at(0)._constructor;
    ASSERT_TRUE(originalConstructor.has_value());
    ASSERT_TRUE(actualConstructor.has_value());
    EXPECT_NE(originalConstructor->_separation, actualConstructor->_separation);
    EXPECT_NE(originalConstructor->_provideEnergy, actualConstructor->_provideEnergy);
}

TEST_F(ConstructorMutationTests, constructorMutation_keepsValidGeneIndex)
{
    auto genome = createTestGenome();
    genome._mutationRates._constructorMutations[0] = ConstructorMutationDesc().eventProbability(1.0f).sigma(1.0f).probability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    auto numGenes = static_cast<int>(actualGenome._genes.size());
    for (auto const& gene : actualGenome._genes) {
        for (auto const& node : gene._nodes) {
            if (node._constructor.has_value()) {
                EXPECT_GE(node._constructor->_geneIndex, 0);
                EXPECT_LT(node._constructor->_geneIndex, numGenes);
                EXPECT_GE(node._constructor->_numBranches, 1);
                EXPECT_LE(node._constructor->_numBranches, 6);
                EXPECT_GE(node._constructor->_reservedEnergy, 0.0f);
                EXPECT_GE(node._constructor->_numConcatenations, 1);
            }
        }
    }
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
