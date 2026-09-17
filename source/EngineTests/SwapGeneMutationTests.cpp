#include <algorithm>

#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Descs.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class SwapGeneMutationTests : public MutationTestsBase
{};

TEST_F(SwapGeneMutationTests, swapGeneMutation_swapsTwoGenes)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().color(1)}),
        GeneDesc().nodes({NodeDesc().color(2), NodeDesc().color(2)}),
    });
    genome._mutationRates._swapGeneMutation = SwapGeneMutationDesc().geneProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(2, actualGenome._genes.size());
    EXPECT_EQ(genome._genes.at(1), actualGenome._genes.at(0));
    EXPECT_EQ(genome._genes.at(0), actualGenome._genes.at(1));
}

TEST_F(SwapGeneMutationTests, swapGeneMutation_referenceKeepsItsGeneIndex)
{
    // The swap exchanges the content behind the gene indices, so the constructor still points to gene 1 and now builds the gene
    // that has moved there.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().nodes({NodeDesc(), NodeDesc()}),
    });
    genome._mutationRates._swapGeneMutation = SwapGeneMutationDesc().geneProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(2, actualGenome._genes.size());
    EXPECT_EQ(2, actualGenome._genes.at(0)._nodes.size());
    ASSERT_EQ(1, actualGenome._genes.at(1)._nodes.size());
    auto const& constructor = actualGenome._genes.at(1)._nodes.at(0)._constructor;
    ASSERT_TRUE(constructor.has_value());
    EXPECT_EQ(1, constructor->_geneIndex);
}

TEST_F(SwapGeneMutationTests, swapGeneMutation_permutesGenesWithoutLoss)
{
    // Every gene is a candidate, but a gene takes part in at most one exchange, so the genome is permuted and no gene is lost or
    // duplicated.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().color(1)}),
        GeneDesc().nodes({NodeDesc().color(2), NodeDesc().color(2)}),
        GeneDesc().nodes({NodeDesc().color(3), NodeDesc().color(3), NodeDesc().color(3)}),
        GeneDesc().nodes({NodeDesc().color(4), NodeDesc().color(4), NodeDesc().color(4), NodeDesc().color(4)}),
    });
    genome._mutationRates._swapGeneMutation = SwapGeneMutationDesc().geneProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(4, actualGenome._genes.size());
    EXPECT_NE(genome._genes, actualGenome._genes);
    EXPECT_TRUE(std::is_permutation(genome._genes.begin(), genome._genes.end(), actualGenome._genes.begin()));
}

TEST_F(SwapGeneMutationTests, swapGeneMutation_singleGeneNoChange)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    genome._mutationRates._swapGeneMutation = SwapGeneMutationDesc().geneProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}

TEST_F(SwapGeneMutationTests, swapGeneMutation_zeroProbabilityNoChange)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().color(1)}),
        GeneDesc().nodes({NodeDesc().color(2), NodeDesc().color(2)}),
    });
    genome._mutationRates._swapGeneMutation = SwapGeneMutationDesc().geneProbability(0.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}
