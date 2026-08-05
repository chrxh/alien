#include <gtest/gtest.h>

#include <EngineInterface/Descs.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class SeparationLimitTests : public MutationTestsBase
{
protected:
    // The separation flag of the n-th node of the given gene; the node must have a constructor.
    bool getSeparation(GenomeDesc const& genome, size_t geneIndex, size_t nodeIndex = 0) const
    {
        auto const& constructor = genome._genes.at(geneIndex)._nodes.at(nodeIndex)._constructor;
        EXPECT_TRUE(constructor.has_value());
        return constructor->_separation;
    }

    // The constructor target of the n-th node of the given gene, or nullopt if that node has no constructor.
    std::optional<int> getConstructorTarget(GenomeDesc const& genome, size_t geneIndex, size_t nodeIndex = 0) const
    {
        auto const& constructor = genome._genes.at(geneIndex)._nodes.at(nodeIndex)._constructor;
        return constructor.has_value() ? std::make_optional(constructor->_geneIndex) : std::nullopt;
    }

    NodeDesc createConstructorNode(int geneIndex, bool separation) const
    {
        return NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(geneIndex).separation(separation));
    }
};

TEST_F(SeparationLimitTests, keepsSingleGeneWithSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1, true)}),
        GeneDesc().nodes({NodeDesc()}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_limitGenesWithSeparation(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(SeparationLimitTests, keepsTwoGenesWithSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1, true), createConstructorNode(2, true)}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_limitGenesWithSeparation(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(SeparationLimitTests, removesSeparationOnThirdGene)
{
    // Losing the separation only means the gene is built within the same creature, the constructor itself stays intact.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1, true), createConstructorNode(2, true), createConstructorNode(3, true)}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_limitGenesWithSeparation(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(4, actualGenome._genes.size());
    EXPECT_TRUE(getSeparation(actualGenome, 0, 0));
    EXPECT_TRUE(getSeparation(actualGenome, 0, 1));
    EXPECT_FALSE(getSeparation(actualGenome, 0, 2));
    EXPECT_EQ(3, getConstructorTarget(actualGenome, 0, 2));
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(SeparationLimitTests, keepsSeveralConstructorsReferencingTheSameGenes)
{
    // Only the number of different genes is limited, so any number of constructors may reference the two accepted genes.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1, true), createConstructorNode(2, true), createConstructorNode(1, true)}),
        GeneDesc().nodes({createConstructorNode(2, true)}),
        GeneDesc().nodes({createConstructorNode(1, true)}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_limitGenesWithSeparation(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(SeparationLimitTests, ignoresConstructorsWithoutSeparation)
{
    // Constructors without separation do not consume one of the two slots, so the separation on gene3 is kept.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1, false), createConstructorNode(2, false), createConstructorNode(3, true)}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_limitGenesWithSeparation(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(SeparationLimitTests, collectsGenesAlongDepthFirstSearch)
{
    // The search descends gene0 -> gene1 -> gene2 before it returns to the second node of gene0, so gene1 and gene2 fill both
    // slots and the reference to gene3 loses its separation.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1, true), createConstructorNode(3, true)}),
        GeneDesc().nodes({createConstructorNode(2, true)}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_limitGenesWithSeparation(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(4, actualGenome._genes.size());
    EXPECT_TRUE(getSeparation(actualGenome, 0, 0));
    EXPECT_FALSE(getSeparation(actualGenome, 0, 1));
    EXPECT_TRUE(getSeparation(actualGenome, 1, 0));
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(SeparationLimitTests, countsRootGeneAsGeneWithSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(0, true), createConstructorNode(1, true), createConstructorNode(2, true)}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_limitGenesWithSeparation(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(3, actualGenome._genes.size());
    EXPECT_TRUE(getSeparation(actualGenome, 0, 0));
    EXPECT_TRUE(getSeparation(actualGenome, 0, 1));
    EXPECT_FALSE(getSeparation(actualGenome, 0, 2));
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}
