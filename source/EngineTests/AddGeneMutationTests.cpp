#include <variant>

#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Descs.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class AddGeneMutationTests : public MutationTestsBase
{};

namespace
{
    int countConstructorsPointingToGene(GeneDesc const& gene, int geneIndex)
    {
        int result = 0;
        for (auto const& node : gene._nodes) {
            if (node._constructor.has_value() && node._constructor->_geneIndex == geneIndex) {
                ++result;
            }
        }
        return result;
    }
}

TEST_F(AddGeneMutationTests, addGeneMutation_appendsSingleNodeGeneAndPointsToIt)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    genome._mutationRates._addGeneMutation = AddGeneMutationDesc().geneProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(2, actualGenome._genes.size());
    EXPECT_EQ(1, actualGenome._genes.at(1)._nodes.size());
    EXPECT_EQ(1, countConstructorsPointingToGene(actualGenome._genes.at(0), 1));
}

TEST_F(AddGeneMutationTests, addGeneMutation_newNodeInheritsCustomizationOfConstructor)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().color(3)})});
    genome._mutationRates._addGeneMutation = AddGeneMutationDesc().geneProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(2, actualGenome._genes.size());
    EXPECT_EQ(3, actualGenome._genes.at(1)._nodes.at(0)._color);
}

TEST_F(AddGeneMutationTests, addGeneMutation_everyGeneGetsItsOwnNewGene)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });
    genome._mutationRates._addGeneMutation = AddGeneMutationDesc().geneProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(4, actualGenome._genes.size());
    EXPECT_EQ(1, actualGenome._genes.at(2)._nodes.size());
    EXPECT_EQ(1, actualGenome._genes.at(3)._nodes.size());

    // Each of the two new genes is referenced exactly once by one of the two original genes.
    auto numReferences = countConstructorsPointingToGene(actualGenome._genes.at(0), 2) + countConstructorsPointingToGene(actualGenome._genes.at(0), 3)
        + countConstructorsPointingToGene(actualGenome._genes.at(1), 2) + countConstructorsPointingToGene(actualGenome._genes.at(1), 3);
    EXPECT_EQ(2, numReferences);
}

TEST_F(AddGeneMutationTests, addGeneMutation_voidNodeDoesNotBecomeConstructor)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().cellType(VoidGenomeDesc())})});
    genome._mutationRates._addGeneMutation = AddGeneMutationDesc().geneProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(1, actualGenome._genes.size());
}

TEST_F(AddGeneMutationTests, addGeneMutation_zeroProbabilityNoChange)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    genome._mutationRates._addGeneMutation = AddGeneMutationDesc().geneProbability(0.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}
