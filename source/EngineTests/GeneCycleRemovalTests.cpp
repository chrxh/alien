#include <gtest/gtest.h>

#include <EngineInterface/Descs.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class GeneCycleRemovalTests : public MutationTestsBase
{
protected:
    // The constructor target of the n-th node of the given gene, or nullopt if that node has no constructor.
    std::optional<int> getConstructorTarget(GenomeDesc const& genome, size_t geneIndex, size_t nodeIndex = 0) const
    {
        auto const& constructor = genome._genes.at(geneIndex)._nodes.at(nodeIndex)._constructor;
        return constructor.has_value() ? std::make_optional(constructor->_geneIndex) : std::nullopt;
    }
};

TEST_F(GeneCycleRemovalTests, breaksSelfReferencingGene)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeGeneCycles(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(2, actualGenome._genes.size());
    EXPECT_EQ(1, getConstructorTarget(actualGenome, 0));
    EXPECT_EQ(std::nullopt, getConstructorTarget(actualGenome, 1));
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(GeneCycleRemovalTests, breaksCycleBetweenTwoGenes)
{
    // Only the constructor closing the cycle (gene2 -> gene1) is removed, the chain gene0 -> gene1 -> gene2 stays intact.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2))}),
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeGeneCycles(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(3, actualGenome._genes.size());
    EXPECT_EQ(1, getConstructorTarget(actualGenome, 0));
    EXPECT_EQ(2, getConstructorTarget(actualGenome, 1));
    EXPECT_EQ(std::nullopt, getConstructorTarget(actualGenome, 2));
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(GeneCycleRemovalTests, breaksSeveralIndependentCycles)
{
    // gene1 <-> gene2 and gene3 <-> gene4 are two cycles not reachable from the root gene; both must be broken.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2))}),
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(4))}),
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3))}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeGeneCycles(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(5, actualGenome._genes.size());
    EXPECT_EQ(std::nullopt, getConstructorTarget(actualGenome, 0));
    EXPECT_EQ(2, getConstructorTarget(actualGenome, 1));
    EXPECT_EQ(std::nullopt, getConstructorTarget(actualGenome, 2));
    EXPECT_EQ(4, getConstructorTarget(actualGenome, 3));
    EXPECT_EQ(std::nullopt, getConstructorTarget(actualGenome, 4));
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(GeneCycleRemovalTests, keepsCycleThroughRootGene)
{
    // Constructing the root gene starts a new creature, so a cycle running through it is intended.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2))}),
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0))}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeGeneCycles(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(GeneCycleRemovalTests, keepsSelfReferencingRootGene)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0))}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeGeneCycles(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(GeneCycleRemovalTests, keepsDiamondShapedReferences)
{
    // gene0 -> gene1 -> gene3 and gene0 -> gene2 -> gene3: gene3 is reached twice without any cycle, so nothing is removed.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2)),
        }),
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3))}),
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3))}),
        GeneDesc().nodes({NodeDesc()}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeGeneCycles(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(GeneCycleRemovalTests, keepsCycleFormedByInjectorReference)
{
    // An injector's gene reference is a payload for a foreign creature, not part of this creature's own construction graph,
    // so it cannot form a cycle.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().nodes({NodeDesc().cellType(InjectorGenomeDesc().geneIndex(1))}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_removeGeneCycles(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}
