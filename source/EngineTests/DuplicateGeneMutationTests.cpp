#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class DuplicateGeneMutationTests : public MutationTestsBase
{
protected:
    void alwaysMutate()
    {
        _parameters.genomeMutationProbability.value = 1.0f;
        _simulationFacade->setSimulationParameters(_parameters);
    }
};

namespace
{
    int countConstructorsReferencingGene(GenomeDesc const& genome, int geneIndex)
    {
        int result = 0;
        for (auto const& gene : genome._genes) {
            for (auto const& node : gene._nodes) {
                if (node._constructor.has_value() && node._constructor->_geneIndex == geneIndex) {
                    ++result;
                }
            }
        }
        return result;
    }
}

TEST_F(DuplicateGeneMutationTests, duplicateGeneMutation_duplicatesGeneReferencedMoreThanOnce)
{
    // Both genes are referenced by two constructors, so whichever gene is picked is duplicated as the new last gene (index 2)
    // and exactly one referencing constructor is repointed to that copy.
    auto genome = GenomeDesc().genes(
        {GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)), NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
         GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0)), NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0))})});
    genome._mutationRates._duplicateGeneMutation = DuplicateGeneMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    alwaysMutate();
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(3, actualGenome._genes.size());
    EXPECT_EQ(2, actualGenome._genes.at(2)._nodes.size());            // the appended copy has the same nodes as the original
    EXPECT_EQ(1, countConstructorsReferencingGene(actualGenome, 2));  // exactly one reference moved to the copy
}

TEST_F(DuplicateGeneMutationTests, duplicateGeneMutation_singleReferenceNoChange)
{
    // A gene referenced only once is not duplicated.
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0)), NodeDesc()})});
    genome._mutationRates._duplicateGeneMutation = DuplicateGeneMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    alwaysMutate();
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(1, actualGenome._genes.size());
}

TEST_F(DuplicateGeneMutationTests, duplicateGeneMutation_zeroProbabilityNoChange)
{
    auto genome = GenomeDesc().genes(
        {GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0)), NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0))})});
    genome._mutationRates._duplicateGeneMutation = DuplicateGeneMutationDesc().geneProbability(0.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    alwaysMutate();
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}
