#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class UnusedGeneRemovalTests : public MutationTestsBase
{
protected:
    // Gene 0 references itself, gene 1 references gene 2 and gene 2 references gene 1 (via node constructors).
    // Gene 1 and gene 2 are not reachable from the root gene (gene 0).
    GenomeDesc createUnreachableCyclicTestGenome() const
    {
        return GenomeDesc().genes({
            GeneDesc().name("gene0").nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0))}),
            GeneDesc().name("gene1").nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2))}),
            GeneDesc().name("gene2").nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        });
    }

    // Gene 0 references gene 1, gene 1 references gene 2 and gene 2 references back to gene 1: a cycle, but reachable from root.
    GenomeDesc createReachableCyclicTestGenome() const
    {
        return GenomeDesc().genes({
            GeneDesc().name("gene0").nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
            GeneDesc().name("gene1").nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2))}),
            GeneDesc().name("gene2").nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        });
    }
};

TEST_F(UnusedGeneRemovalTests, removeUnreachableGenesFromRoot_removesGenesUnreachableFromRoot)
{
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), createUnreachableCyclicTestGenome());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeUnusedGenes(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(1, actualGenome._genes.size());
    EXPECT_EQ("gene0", actualGenome._genes.at(0)._name);
    EXPECT_EQ(0, actualGenome._genes.at(0)._nodes.at(0)._constructor->_geneIndex);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(UnusedGeneRemovalTests, removeUnreachableGenesFromRoot_keepsCyclicallyReferencedGenesReachableFromRoot)
{
    auto genome = createReachableCyclicTestGenome();
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeUnusedGenes(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}

TEST_F(UnusedGeneRemovalTests, removeUnreachableGenesFromRoot_allGenesReachable_noChange)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().name("gene0").nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().name("gene1").nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2))}),
        GeneDesc().name("gene2").nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeUnusedGenes(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}

TEST_F(UnusedGeneRemovalTests, removeUnreachableGenesFromRoot_keepsGeneReferencedByInjector)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().name("gene0").nodes({NodeDesc().cellType(InjectorGenomeDesc().geneIndex(2))}),
        GeneDesc().name("gene1").nodes({NodeDesc()}),
        GeneDesc().name("gene2").nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeUnusedGenes(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(2, actualGenome._genes.size());
    EXPECT_EQ("gene0", actualGenome._genes.at(0)._name);
    EXPECT_EQ("gene2", actualGenome._genes.at(1)._name);
    EXPECT_EQ(1, std::get<InjectorGenomeDesc>(actualGenome._genes.at(0)._nodes.at(0)._cellType)._geneIndex);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(UnusedGeneRemovalTests, removeUnreachableGenesFromRoot_keepsGeneReferencedByConstructorOnVoidNode)
{
    // A void node's constructor data still counts as a reference and keeps its target gene alive.
    auto genome = GenomeDesc().genes({
        GeneDesc().name("gene0").nodes({NodeDesc().cellType(VoidGenomeDesc()).constructor(ConstructorGenomeDesc().geneIndex(1)), NodeDesc()}),
        GeneDesc().name("gene1").nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeUnusedGenes(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(UnusedGeneRemovalTests, applyMutations_removesGenesUnreachableFromRoot)
{
    // All mutation rates are zero, so applying mutations only removes the genes unreachable from the root gene.
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), createUnreachableCyclicTestGenome());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(1, actualGenome._genes.size());
    EXPECT_EQ("gene0", actualGenome._genes.at(0)._name);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(UnusedGeneRemovalTests, applyMutations_allGenesReachableFromRoot_noChange)
{
    auto genome = createReachableCyclicTestGenome();
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}
