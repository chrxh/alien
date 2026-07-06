#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class UnusedGeneRemovalTests : public MutationTestsBase
{
protected:
    // Gene 0 references gene 0, gene 1 references gene 2 and gene 2 references gene 1 (via node constructors).
    GenomeDesc createCyclicTestGenome() const
    {
        return GenomeDesc().genes({
            GeneDesc().name("gene0").nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0))}),
            GeneDesc().name("gene1").nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2))}),
            GeneDesc().name("gene2").nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        });
    }
};

TEST_F(UnusedGeneRemovalTests, removeUnusedGenes_referenceGene0_removesUnreachableGenes)
{
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), createCyclicTestGenome());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeUnusedGenes(1, 0);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(1, actualGenome._genes.size());
    EXPECT_EQ("gene0", actualGenome._genes.at(0)._name);
    EXPECT_EQ(0, actualGenome._genes.at(0)._nodes.at(0)._constructor->_geneIndex);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(UnusedGeneRemovalTests, removeUnusedGenes_referenceGene1_keepsCyclicallyReferencedGenes)
{
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), createCyclicTestGenome());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeUnusedGenes(1, 1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(2, actualGenome._genes.size());
    EXPECT_EQ("gene1", actualGenome._genes.at(0)._name);
    EXPECT_EQ("gene2", actualGenome._genes.at(1)._name);
    EXPECT_EQ(1, actualGenome._genes.at(0)._nodes.at(0)._constructor->_geneIndex);
    EXPECT_EQ(0, actualGenome._genes.at(1)._nodes.at(0)._constructor->_geneIndex);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(UnusedGeneRemovalTests, removeUnusedGenes_allGenesReachable_noChange)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().name("gene0").nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().name("gene1").nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2))}),
        GeneDesc().name("gene2").nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeUnusedGenes(1, 0);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}

TEST_F(UnusedGeneRemovalTests, removeUnusedGenes_keepsGeneReferencedByInjector)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().name("gene0").nodes({NodeDesc().cellType(InjectorGenomeDesc().geneIndex(2))}),
        GeneDesc().name("gene1").nodes({NodeDesc()}),
        GeneDesc().name("gene2").nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeUnusedGenes(1, 0);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(2, actualGenome._genes.size());
    EXPECT_EQ("gene0", actualGenome._genes.at(0)._name);
    EXPECT_EQ("gene2", actualGenome._genes.at(1)._name);
    EXPECT_EQ(1, std::get<InjectorGenomeDesc>(actualGenome._genes.at(0)._nodes.at(0)._cellType)._geneIndex);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(UnusedGeneRemovalTests, removeUnusedGenes_invalidReferenceGeneIndex_noChange)
{
    auto genome = createCyclicTestGenome();
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeUnusedGenes(1, -1);
    _simulationFacade->testOnly_removeUnusedGenes(1, 3);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}

TEST_F(UnusedGeneRemovalTests, applyMutations_removesUnusedGenes)
{
    // All mutation rates are zero, so applying mutations only removes the genes that are unreachable from the reference gene.
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), createCyclicTestGenome());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1, 1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(2, actualGenome._genes.size());
    EXPECT_EQ("gene1", actualGenome._genes.at(0)._name);
    EXPECT_EQ("gene2", actualGenome._genes.at(1)._name);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(UnusedGeneRemovalTests, applyMutations_negativeReferenceGeneIndex_keepsUnusedGenes)
{
    auto genome = createCyclicTestGenome();
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}
