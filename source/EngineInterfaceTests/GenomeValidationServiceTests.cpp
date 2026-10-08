#include <gtest/gtest.h>

#include <Data/GenomeDesc.h>

#include <EngineInterface/GenomeValidationService.h>

class GenomeValidationServiceTests : public ::testing::Test
{
public:
    GenomeValidationServiceTests() = default;
    virtual ~GenomeValidationServiceTests() = default;

protected:
    std::vector<GenomeIssue> validate(GenomeDesc const& genome) const { return GenomeValidationService::get().validate(genome); }

    NodeDesc createConstructorNode(int geneIndex, bool separation = false) const
    {
        return NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(geneIndex).separation(separation));
    }

    NodeDesc createVoidNode() const { return NodeDesc().cellType(VoidGenomeDesc()); }

    NodeDesc createInjectorNode(int geneIndex) const { return NodeDesc().cellType(InjectorGenomeDesc().geneIndex(geneIndex)); }
};

TEST_F(GenomeValidationServiceTests, validGenome_noIssues)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1), createConstructorNode(0, true)}),
        GeneDesc().nodes({NodeDesc(), NodeDesc()}),
    });

    EXPECT_TRUE(validate(genome).empty());
}

TEST_F(GenomeValidationServiceTests, voidNodeInSegment_removesGene)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1), NodeDesc()}),
        GeneDesc().nodes({NodeDesc(), createVoidNode(), NodeDesc()}),
    });

    auto issues = validate(genome);

    ASSERT_EQ(2, issues.size());
    EXPECT_EQ(GenomeIssueType::NodesCutOffFromLastNode, issues.at(0).type);
    EXPECT_EQ(1, issues.at(0).geneIndex);
    EXPECT_EQ(std::nullopt, issues.at(0).nodeIndex);
    EXPECT_EQ(std::vector{0}, issues.at(0).voidedNodeIndices);
    EXPECT_TRUE(issues.at(0).removesGene);

    EXPECT_EQ(GenomeIssueType::ConstructsRemovedGene, issues.at(1).type);
    EXPECT_EQ(0, issues.at(1).geneIndex);
    EXPECT_EQ(0, issues.at(1).nodeIndex);
    EXPECT_EQ(std::vector{1}, issues.at(1).relatedGeneIndices);
    EXPECT_EQ(0, issues.at(1).causeIssueIndex);
}

TEST_F(GenomeValidationServiceTests, voidNodesInTriangle_voidsNodesCutOffFromLastNode)
{
    // The void nodes 3, 8 and 12 separate the nodes 9, 10 and 11 from the last node, the additional connections of the triangle keep the
    // other nodes connected
    std::vector<NodeDesc> nodes(15);
    for (auto nodeIndex : {3, 8, 12}) {
        nodes.at(nodeIndex).cellType(VoidGenomeDesc());
    }
    auto genome = GenomeDesc().genes({GeneDesc().shape(ConstructorShape_Triangle).nodes(nodes)});

    auto issues = validate(genome);

    ASSERT_EQ(1, issues.size());
    EXPECT_EQ(GenomeIssueType::NodesCutOffFromLastNode, issues.at(0).type);
    EXPECT_EQ((std::vector{9, 10, 11}), issues.at(0).voidedNodeIndices);
    EXPECT_FALSE(issues.at(0).removesGene);
}

TEST_F(GenomeValidationServiceTests, homogeneousCellType_noIssues)
{
    auto genome = GenomeDesc().genes({GeneDesc().homogeneousCellType(true).nodes({NodeDesc(), createVoidNode(), NodeDesc()})});

    EXPECT_TRUE(validate(genome).empty());
}

TEST_F(GenomeValidationServiceTests, onlyGene_isKept)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), createVoidNode(), NodeDesc()})});

    auto issues = validate(genome);

    ASSERT_EQ(1, issues.size());
    EXPECT_EQ(GenomeIssueType::NodesCutOffFromLastNode, issues.at(0).type);
    EXPECT_EQ(std::vector{0}, issues.at(0).voidedNodeIndices);
    EXPECT_FALSE(issues.at(0).removesGene);
}

TEST_F(GenomeValidationServiceTests, voidLastNode_withoutNodeMutations_removesGene)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1), NodeDesc()}),
        GeneDesc().nodes({NodeDesc(), NodeDesc(), createVoidNode()}),
    });

    auto issues = validate(genome);

    ASSERT_EQ(2, issues.size());
    EXPECT_EQ(GenomeIssueType::NodesCutOffFromLastNode, issues.at(0).type);
    EXPECT_EQ(1, issues.at(0).geneIndex);
    EXPECT_EQ(2, issues.at(0).nodeIndex);
    EXPECT_EQ((std::vector{0, 1}), issues.at(0).voidedNodeIndices);
    EXPECT_TRUE(issues.at(0).removesGene);
    EXPECT_EQ(GenomeIssueType::ConstructsRemovedGene, issues.at(1).type);
}

TEST_F(GenomeValidationServiceTests, voidLastNode_withNodeMutations_getsRandomCellType)
{
    auto genome = GenomeDesc()
                      .genes({
                          GeneDesc().nodes({createConstructorNode(1), NodeDesc()}),
                          GeneDesc().nodes({NodeDesc(), NodeDesc(), createVoidNode()}),
                      })
                      .mutationRates(MutationRatesDesc().addNodeMutation(AddNodeMutationDesc().nodeProbability(0.1f)));

    auto issues = validate(genome);

    ASSERT_EQ(1, issues.size());
    EXPECT_EQ(GenomeIssueType::VoidBoundaryNode, issues.at(0).type);
    EXPECT_EQ(1, issues.at(0).geneIndex);
    EXPECT_EQ(2, issues.at(0).nodeIndex);
}

TEST_F(GenomeValidationServiceTests, cycleBetweenTwoGenes)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1)}),
        GeneDesc().nodes({createConstructorNode(2)}),
        GeneDesc().nodes({createConstructorNode(1)}),
    });

    auto issues = validate(genome);

    ASSERT_EQ(1, issues.size());
    EXPECT_EQ(GenomeIssueType::CycleAvoidingRootGene, issues.at(0).type);
    EXPECT_EQ(2, issues.at(0).geneIndex);
    EXPECT_EQ(0, issues.at(0).nodeIndex);
    EXPECT_EQ((std::vector{1, 2, 1}), issues.at(0).relatedGeneIndices);
}

TEST_F(GenomeValidationServiceTests, selfReferencingGene)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1)}),
        GeneDesc().nodes({createConstructorNode(1)}),
    });

    auto issues = validate(genome);

    ASSERT_EQ(1, issues.size());
    EXPECT_EQ(GenomeIssueType::CycleAvoidingRootGene, issues.at(0).type);
    EXPECT_EQ(1, issues.at(0).geneIndex);
    EXPECT_EQ((std::vector{1, 1}), issues.at(0).relatedGeneIndices);
}

TEST_F(GenomeValidationServiceTests, cycleThroughRootGene_noIssues)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1)}),
        GeneDesc().nodes({createConstructorNode(2)}),
        GeneDesc().nodes({createConstructorNode(0)}),
    });

    EXPECT_TRUE(validate(genome).empty());
}

TEST_F(GenomeValidationServiceTests, unreachableGene)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc()}), GeneDesc().nodes({NodeDesc()})});

    auto issues = validate(genome);

    ASSERT_EQ(1, issues.size());
    EXPECT_EQ(GenomeIssueType::GeneUnreachableFromRoot, issues.at(0).type);
    EXPECT_EQ(1, issues.at(0).geneIndex);
    EXPECT_TRUE(issues.at(0).removesGene);
    EXPECT_EQ(std::nullopt, issues.at(0).causeIssueIndex);
}

TEST_F(GenomeValidationServiceTests, unreachableAfterCycleRemoval_isFollowUp)
{
    // The search starts at gene 1 and turns off the constructor of gene 2, which was the only way from the root gene to gene 1
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(2)}),
        GeneDesc().nodes({createConstructorNode(2)}),
        GeneDesc().nodes({createConstructorNode(1)}),
    });

    auto issues = validate(genome);

    ASSERT_EQ(2, issues.size());
    EXPECT_EQ(GenomeIssueType::CycleAvoidingRootGene, issues.at(0).type);
    EXPECT_EQ(2, issues.at(0).geneIndex);
    EXPECT_EQ(GenomeIssueType::GeneUnreachableFromRoot, issues.at(1).type);
    EXPECT_EQ(1, issues.at(1).geneIndex);
    EXPECT_EQ(0, issues.at(1).causeIssueIndex);
}

TEST_F(GenomeValidationServiceTests, tooManyGenesWithSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1, true), createConstructorNode(2, true), createConstructorNode(3, true)}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });

    auto issues = validate(genome);

    ASSERT_EQ(1, issues.size());
    EXPECT_EQ(GenomeIssueType::TooManyGenesWithSeparation, issues.at(0).type);
    EXPECT_EQ(0, issues.at(0).geneIndex);
    EXPECT_EQ(2, issues.at(0).nodeIndex);
    EXPECT_EQ((std::vector{1, 2}), issues.at(0).relatedGeneIndices);
}

TEST_F(GenomeValidationServiceTests, injectorOfRemovedGene_isFollowUp)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({createInjectorNode(1)}), GeneDesc().nodes({NodeDesc()})});

    auto issues = validate(genome);

    ASSERT_EQ(2, issues.size());
    EXPECT_EQ(GenomeIssueType::GeneUnreachableFromRoot, issues.at(0).type);
    EXPECT_EQ(GenomeIssueType::InjectsRemovedGene, issues.at(1).type);
    EXPECT_EQ(0, issues.at(1).geneIndex);
    EXPECT_EQ(0, issues.at(1).nodeIndex);
    EXPECT_EQ(0, issues.at(1).causeIssueIndex);
}

TEST_F(GenomeValidationServiceTests, combinedIssues)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(0, true), NodeDesc(), createConstructorNode(1), NodeDesc(), createConstructorNode(2), NodeDesc()}),
        GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc(), createConstructorNode(3), createConstructorNode(4, true)}),
        GeneDesc().nodes({NodeDesc(), NodeDesc(), createVoidNode(), NodeDesc(), NodeDesc()}),
        GeneDesc().shape(ConstructorShape_Triangle).nodes({NodeDesc(), createConstructorNode(1), NodeDesc(), createConstructorNode(5, true)}),
        GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc()}),
        GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc()}),
        GeneDesc().nodes({NodeDesc(), NodeDesc()}),
    });

    auto issues = validate(genome);

    ASSERT_EQ(5, issues.size());
    EXPECT_EQ(GenomeIssueType::NodesCutOffFromLastNode, issues.at(0).type);
    EXPECT_EQ(2, issues.at(0).geneIndex);
    EXPECT_EQ(GenomeIssueType::ConstructsRemovedGene, issues.at(1).type);
    EXPECT_EQ(0, issues.at(1).geneIndex);
    EXPECT_EQ(4, issues.at(1).nodeIndex);
    EXPECT_EQ(GenomeIssueType::CycleAvoidingRootGene, issues.at(2).type);
    EXPECT_EQ(3, issues.at(2).geneIndex);
    EXPECT_EQ((std::vector{1, 3, 1}), issues.at(2).relatedGeneIndices);
    EXPECT_EQ(GenomeIssueType::GeneUnreachableFromRoot, issues.at(3).type);
    EXPECT_EQ(6, issues.at(3).geneIndex);
    EXPECT_EQ(GenomeIssueType::TooManyGenesWithSeparation, issues.at(4).type);
    EXPECT_EQ(1, issues.at(4).geneIndex);
    EXPECT_EQ(4, issues.at(4).nodeIndex);
    EXPECT_EQ((std::vector{0, 5}), issues.at(4).relatedGeneIndices);
}

TEST_F(GenomeValidationServiceTests, correct_removesGenesAndRemapsReferences)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1), createConstructorNode(3), createInjectorNode(2)}),
        GeneDesc().nodes({NodeDesc(), createVoidNode(), NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });

    auto newGeneIndices = GenomeValidationService::get().correct(genome);

    EXPECT_EQ((std::vector{0, -1, -1, 1}), newGeneIndices);
    auto expectedGenes = std::vector{
        GeneDesc().nodes({NodeDesc(), createConstructorNode(1), createInjectorNode(0)}),
        GeneDesc().nodes({NodeDesc()}),
    };
    EXPECT_EQ(expectedGenes, genome._genes);
    EXPECT_TRUE(validate(genome).empty());
}

TEST_F(GenomeValidationServiceTests, correct_turnsOffConstructorAndSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1, true), createConstructorNode(2, true), createConstructorNode(3, true)}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({createConstructorNode(3)}),
    });

    GenomeValidationService::get().correct(genome);

    auto const& rootNodes = genome._genes.at(0)._nodes;
    EXPECT_TRUE(rootNodes.at(0)._constructor->_separation);
    EXPECT_TRUE(rootNodes.at(1)._constructor->_separation);
    EXPECT_FALSE(rootNodes.at(2)._constructor->_separation);
    EXPECT_EQ(std::nullopt, genome._genes.at(3)._nodes.at(0)._constructor);
    EXPECT_TRUE(validate(genome).empty());
}

TEST_F(GenomeValidationServiceTests, correct_voidBoundaryNodeBecomesBaseCell)
{
    auto genome = GenomeDesc()
                      .genes({GeneDesc().nodes({NodeDesc(), NodeDesc().cellType(SensorGenomeDesc()), createVoidNode()})})
                      .mutationRates(MutationRatesDesc().deleteNodeMutation(DeleteNodeMutationDesc().nodeProbability(0.1f)));

    GenomeValidationService::get().correct(genome);

    EXPECT_EQ(CellType_Base, genome._genes.at(0)._nodes.at(2).getCellType());
    EXPECT_TRUE(validate(genome).empty());
}
