#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class DeleteGeneMutationTests : public MutationTestsBase
{};

TEST_F(DeleteGeneMutationTests, deleteGeneMutation_deletesAllGenesIncludingRoot)
{
    // With a probability of 1 every gene (including the root gene) is deleted, leaving an empty genome.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });
    genome._mutationRates._deleteGeneMutation = DeleteGeneMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(0, actualGenome._genes.size());
}

TEST_F(DeleteGeneMutationTests, deleteGeneMutation_zeroProbabilityNoChange)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().nodes({NodeDesc()}),
    });
    genome._mutationRates._deleteGeneMutation = DeleteGeneMutationDesc().geneProbability(0.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}
