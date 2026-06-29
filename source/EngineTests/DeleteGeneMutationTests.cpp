#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class DeleteGeneMutationTests : public MutationTestsBase
{};

TEST_F(DeleteGeneMutationTests, deleteGeneMutation_keepsAtLeastOneGene)
{
    // Even with a probability of 1, at least one gene must survive so the genome never becomes empty.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });
    genome._mutationRates._deleteGeneMutation = DeleteGeneMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 5; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(1, actualGenome._genes.size());
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
