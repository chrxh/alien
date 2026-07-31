#include <gtest/gtest.h>

#include <EngineInterface/Descs.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class UnreachableNodeVoidingTests : public MutationTestsBase
{};

TEST_F(UnreachableNodeVoidingTests, noVoidNodes_noChange)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc()}),
        GeneDesc().shape(ConstructorShape_Triangle).nodes({NodeDesc(), NodeDesc(), NodeDesc(), NodeDesc()}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_voidUnreachableNodes(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(UnreachableNodeVoidingTests, voidsNodesConnectedOnlyViaVoidNodes)
{
    // A triangle with 15 nodes, where the void nodes 3, 8 and 12 separate the nodes 9, 10 and 11 from the last node: their only
    // connections left are the ones among each other. The nodes 0 to 7 stay connected to the last node via the additional
    // connections of the nodes 13 and 14.
    std::vector<NodeDesc> nodes(15);
    for (auto nodeIndex : {3, 8, 12}) {
        nodes.at(nodeIndex).cellType(VoidGenomeDesc());
    }
    auto genome = GenomeDesc().genes({GeneDesc().shape(ConstructorShape_Triangle).nodes(nodes)});
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_voidUnreachableNodes(1);

    auto expectedGenome = genome;
    for (auto nodeIndex : {9, 10, 11}) {
        expectedGenome._genes.at(0)._nodes.at(nodeIndex).cellType(VoidGenomeDesc());
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(expectedGenome, actualGenome);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(UnreachableNodeVoidingTests, voidedNodeLosesItsConstructor)
{
    auto genome = GenomeDesc().genes({
        GeneDesc()
            .shape(ConstructorShape_Triangle)
            .nodes({
                NodeDesc(),
                NodeDesc(),
                NodeDesc().cellType(VoidGenomeDesc()),
                NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)),
                NodeDesc().cellType(VoidGenomeDesc()),
                NodeDesc(),
            }),
        GeneDesc().nodes({NodeDesc()}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_voidUnreachableNodes(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(2, actualGenome._genes.size());
    auto const& actualNode = actualGenome._genes.at(0)._nodes.at(3);
    EXPECT_EQ(CellType_Void, actualNode.getCellType());
    EXPECT_EQ(std::nullopt, actualNode._constructor);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(UnreachableNodeVoidingTests, removesGeneWithVoidedFirstNode)
{
    // A void node in a segment separates all preceding nodes from the last node, so the first node of gene1 is voided as well
    // and the whole gene is removed. The constructor referencing it is turned off.
    auto genome = GenomeDesc().genes({
        GeneDesc().name("gene0").nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)), NodeDesc()}),
        GeneDesc().name("gene1").nodes({NodeDesc(), NodeDesc().cellType(VoidGenomeDesc()), NodeDesc()}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_voidUnreachableNodes(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(1, actualGenome._genes.size());
    EXPECT_EQ("gene0", actualGenome._genes.at(0)._name);
    EXPECT_EQ(std::nullopt, actualGenome._genes.at(0)._nodes.at(0)._constructor);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(UnreachableNodeVoidingTests, keepsLastGeneOfGenome)
{
    // The genome must not become empty, so the only gene survives even though its first node is voided.
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc().cellType(VoidGenomeDesc()), NodeDesc()})});
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_voidUnreachableNodes(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(1, actualGenome._genes.size());
    auto const& actualNodes = actualGenome._genes.at(0)._nodes;
    ASSERT_EQ(3, actualNodes.size());
    EXPECT_EQ(CellType_Void, actualNodes.at(0).getCellType());
    EXPECT_NE(CellType_Void, actualNodes.at(2).getCellType());
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}

TEST_F(UnreachableNodeVoidingTests, homogeneousCellType_noChange)
{
    // With a homogeneous cell type the effective cell type of every node is taken from the first node, so there is no void node
    // that could separate the gene.
    auto genome = GenomeDesc().genes({
        GeneDesc().homogeneousCellType(true).nodes({NodeDesc(), NodeDesc().cellType(VoidGenomeDesc()), NodeDesc()}),
    });
    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_voidUnreachableNodes(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
}
