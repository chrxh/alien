#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class DeleteGeneMutationTests : public MutationTestsBase
{
protected:
    void alwaysMutate()
    {
        _parameters.genomeMutationProbability.value = 1.0f;
        _simulationFacade->setSimulationParameters(_parameters);
    }
};

TEST_F(DeleteGeneMutationTests, deleteGeneMutation_deletesGenesKeepingAtLeastOne)
{
    // With a probability of 1 every gene is marked for deletion, but at least one gene is always kept.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });
    genome._mutationRates._deleteGeneMutation = DeleteGeneMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    alwaysMutate();
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(1, actualGenome._genes.size());
}

TEST_F(DeleteGeneMutationTests, deleteGeneMutation_deletesRootGeneAndDisablesReferencingConstructor)
{
    // The root gene (index 0) can be deleted too; the surviving constructor that referenced it is turned off.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0))}),
    });
    genome._mutationRates._deleteGeneMutation = DeleteGeneMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    alwaysMutate();
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(1, actualGenome._genes.size());
    EXPECT_FALSE(actualGenome._genes.at(0)._nodes.at(0)._constructor.has_value());
}

TEST_F(DeleteGeneMutationTests, deleteGeneMutation_zeroProbabilityNoChange)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().nodes({NodeDesc()}),
    });
    genome._mutationRates._deleteGeneMutation = DeleteGeneMutationDesc().geneProbability(0.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    alwaysMutate();
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}
