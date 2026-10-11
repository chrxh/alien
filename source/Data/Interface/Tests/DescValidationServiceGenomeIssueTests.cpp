#include <random>
#include <ranges>

#include <gtest/gtest.h>

#include <Data/Interface/DescValidationService.h>
#include <Data/Interface/GenomeDesc.h>

#include <Engine/Interface/TestData/DescTestDataFactory.h>

class DescValidationServiceGenomeIssueTests : public ::testing::Test
{
public:
    DescValidationServiceGenomeIssueTests() = default;
    virtual ~DescValidationServiceGenomeIssueTests() = default;

protected:
    std::vector<GenomeIssue> findIssues(GenomeDesc const& genome) const { return DescValidationService::get().findGenomeIssues(genome); }

    NodeDesc createConstructorNode(int geneIndex, bool separation = false) const
    {
        return NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(geneIndex).separation(separation));
    }

    NodeDesc createVoidNode() const { return NodeDesc().cellType(VoidGenomeDesc()); }

    NodeDesc createInjectorNode(int geneIndex) const { return NodeDesc().cellType(InjectorGenomeDesc().geneIndex(geneIndex)); }

    MutationRatesDesc createStructuralNodeMutationRates() const { return MutationRatesDesc().addNodeMutation(AddNodeMutationDesc().nodeProbability(0.1f)); }

    GenomeDesc createTriangleWithVoidNodes3_8_12() const
    {
        std::vector<NodeDesc> nodes(15);
        for (auto nodeIndex : {3, 8, 12}) {
            nodes.at(nodeIndex).cellType(VoidGenomeDesc());
        }
        return GenomeDesc().genes({GeneDesc().shape(ConstructorShape_Triangle).nodes(nodes)});
    }
};

TEST_F(DescValidationServiceGenomeIssueTests, validGenome_noIssues)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1), createConstructorNode(0, true)}),
        GeneDesc().nodes({NodeDesc(), NodeDesc()}),
    });

    EXPECT_TRUE(findIssues(genome).empty());
}

TEST_F(DescValidationServiceGenomeIssueTests, voidNodeInSegment_removesGene)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1), NodeDesc()}),
        GeneDesc().nodes({NodeDesc(), createVoidNode(), NodeDesc()}),
    });

    auto issues = findIssues(genome);

    ASSERT_EQ(2, issues.size());
    EXPECT_EQ(GenomeIssueType::NodesCutOffFromLastNode, issues.at(0).type);
    EXPECT_EQ(1, issues.at(0).geneIndex);
    EXPECT_EQ(std::nullopt, issues.at(0).nodeIndex);
    EXPECT_EQ((std::vector{0, 1}), issues.at(0).cutOffNodeIndices);
    EXPECT_TRUE(issues.at(0).removesGene);
    EXPECT_FALSE(issues.at(0).isFollowUp());

    EXPECT_EQ(GenomeIssueType::ConstructsRemovedGene, issues.at(1).type);
    EXPECT_EQ(0, issues.at(1).geneIndex);
    EXPECT_EQ(0, issues.at(1).nodeIndex);
    EXPECT_EQ(1, issues.at(1).referencedGeneIndex);
    EXPECT_EQ(0, issues.at(1).causeIssueIndex);
}

TEST_F(DescValidationServiceGenomeIssueTests, voidNodesInTriangle_cutOffNodes)
{
    auto issues = findIssues(createTriangleWithVoidNodes3_8_12());

    ASSERT_EQ(1, issues.size());
    EXPECT_EQ(GenomeIssueType::NodesCutOffFromLastNode, issues.at(0).type);
    EXPECT_EQ((std::vector{3, 8, 9, 10, 11, 12}), issues.at(0).cutOffNodeIndices);
    EXPECT_FALSE(issues.at(0).removesGene);
}

TEST_F(DescValidationServiceGenomeIssueTests, homogeneousCellType_noIssues)
{
    auto genome = GenomeDesc().genes({GeneDesc().homogeneousCellType(true).nodes({NodeDesc(), createVoidNode(), NodeDesc()})});

    EXPECT_TRUE(findIssues(genome).empty());
}

TEST_F(DescValidationServiceGenomeIssueTests, onlyGene_isKept)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), createVoidNode(), NodeDesc()})});

    auto issues = findIssues(genome);

    ASSERT_EQ(1, issues.size());
    EXPECT_EQ(GenomeIssueType::NodesCutOffFromLastNode, issues.at(0).type);
    EXPECT_EQ((std::vector{0, 1}), issues.at(0).cutOffNodeIndices);
    EXPECT_FALSE(issues.at(0).removesGene);
}

TEST_F(DescValidationServiceGenomeIssueTests, voidLastNode_withoutNodeMutations_removesGene)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1), NodeDesc()}),
        GeneDesc().nodes({NodeDesc(), NodeDesc(), createVoidNode()}),
    });

    auto issues = findIssues(genome);

    ASSERT_EQ(2, issues.size());
    EXPECT_EQ(GenomeIssueType::NodesCutOffFromLastNode, issues.at(0).type);
    EXPECT_EQ(1, issues.at(0).geneIndex);
    EXPECT_EQ(2, issues.at(0).nodeIndex);
    EXPECT_EQ((std::vector{0, 1}), issues.at(0).cutOffNodeIndices);
    EXPECT_TRUE(issues.at(0).removesGene);
    EXPECT_EQ(GenomeIssueType::ConstructsRemovedGene, issues.at(1).type);
}

TEST_F(DescValidationServiceGenomeIssueTests, voidLastNode_withNodeMutations_getsRandomCellType)
{
    auto genome = GenomeDesc()
                      .genes({
                          GeneDesc().nodes({createConstructorNode(1), NodeDesc()}),
                          GeneDesc().nodes({NodeDesc(), NodeDesc(), createVoidNode()}),
                      })
                      .mutationRates(createStructuralNodeMutationRates());

    auto issues = findIssues(genome);

    ASSERT_EQ(1, issues.size());
    EXPECT_EQ(GenomeIssueType::VoidBoundaryNode, issues.at(0).type);
    EXPECT_EQ(1, issues.at(0).geneIndex);
    EXPECT_EQ(2, issues.at(0).nodeIndex);
}

TEST_F(DescValidationServiceGenomeIssueTests, voidFirstNodeCutOffAgain_withNodeMutations_noVoidBoundaryIssue)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({createVoidNode(), createVoidNode(), NodeDesc()})}).mutationRates(createStructuralNodeMutationRates());

    EXPECT_TRUE(findIssues(genome).empty());
}

TEST_F(DescValidationServiceGenomeIssueTests, cycleBetweenTwoGenes)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1)}),
        GeneDesc().nodes({createConstructorNode(2)}),
        GeneDesc().nodes({createConstructorNode(1)}),
    });

    auto issues = findIssues(genome);

    ASSERT_EQ(1, issues.size());
    EXPECT_EQ(GenomeIssueType::CycleAvoidingRootGene, issues.at(0).type);
    EXPECT_EQ(2, issues.at(0).geneIndex);
    EXPECT_EQ(0, issues.at(0).nodeIndex);
    EXPECT_EQ((std::vector{1, 2, 1}), issues.at(0).cycleGeneIndices);
}

TEST_F(DescValidationServiceGenomeIssueTests, selfReferencingGene)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1)}),
        GeneDesc().nodes({createConstructorNode(1)}),
    });

    auto issues = findIssues(genome);

    ASSERT_EQ(1, issues.size());
    EXPECT_EQ(GenomeIssueType::CycleAvoidingRootGene, issues.at(0).type);
    EXPECT_EQ(1, issues.at(0).geneIndex);
    EXPECT_EQ((std::vector{1, 1}), issues.at(0).cycleGeneIndices);
}

TEST_F(DescValidationServiceGenomeIssueTests, cycleThroughRootGene_noIssues)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1)}),
        GeneDesc().nodes({createConstructorNode(2)}),
        GeneDesc().nodes({createConstructorNode(0)}),
    });

    EXPECT_TRUE(findIssues(genome).empty());
}

TEST_F(DescValidationServiceGenomeIssueTests, unreachableGene)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc()}), GeneDesc().nodes({NodeDesc()})});

    auto issues = findIssues(genome);

    ASSERT_EQ(1, issues.size());
    EXPECT_EQ(GenomeIssueType::GeneUnreachableFromRoot, issues.at(0).type);
    EXPECT_EQ(1, issues.at(0).geneIndex);
    EXPECT_TRUE(issues.at(0).removesGene);
    EXPECT_FALSE(issues.at(0).isFollowUp());
}

TEST_F(DescValidationServiceGenomeIssueTests, unreachableAfterCycleRemoval_isFollowUp)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(2)}),
        GeneDesc().nodes({createConstructorNode(2)}),
        GeneDesc().nodes({createConstructorNode(1)}),
    });

    auto issues = findIssues(genome);

    ASSERT_EQ(2, issues.size());
    EXPECT_EQ(GenomeIssueType::CycleAvoidingRootGene, issues.at(0).type);
    EXPECT_EQ(2, issues.at(0).geneIndex);
    EXPECT_EQ(GenomeIssueType::GeneUnreachableFromRoot, issues.at(1).type);
    EXPECT_EQ(1, issues.at(1).geneIndex);
    EXPECT_EQ(0, issues.at(1).causeIssueIndex);
}

TEST_F(DescValidationServiceGenomeIssueTests, tooManyGenesWithSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1, true), createConstructorNode(2, true), createConstructorNode(3, true)}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });

    auto issues = findIssues(genome);

    ASSERT_EQ(1, issues.size());
    EXPECT_EQ(GenomeIssueType::TooManyGenesWithSeparation, issues.at(0).type);
    EXPECT_EQ(0, issues.at(0).geneIndex);
    EXPECT_EQ(2, issues.at(0).nodeIndex);
    EXPECT_EQ((std::vector{1, 2}), issues.at(0).geneIndicesKeepingSeparation);
}

TEST_F(DescValidationServiceGenomeIssueTests, injectorOfRemovedGene_isFollowUp)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({createInjectorNode(1)}), GeneDesc().nodes({NodeDesc()})});

    auto issues = findIssues(genome);

    ASSERT_EQ(2, issues.size());
    EXPECT_EQ(GenomeIssueType::GeneUnreachableFromRoot, issues.at(0).type);
    EXPECT_EQ(GenomeIssueType::InjectsRemovedGene, issues.at(1).type);
    EXPECT_EQ(0, issues.at(1).geneIndex);
    EXPECT_EQ(0, issues.at(1).nodeIndex);
    EXPECT_EQ(0, issues.at(1).causeIssueIndex);
}

TEST_F(DescValidationServiceGenomeIssueTests, combinedIssues)
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

    auto issues = findIssues(genome);

    ASSERT_EQ(5, issues.size());
    EXPECT_EQ(GenomeIssueType::NodesCutOffFromLastNode, issues.at(0).type);
    EXPECT_EQ(2, issues.at(0).geneIndex);
    EXPECT_EQ(GenomeIssueType::ConstructsRemovedGene, issues.at(1).type);
    EXPECT_EQ(0, issues.at(1).geneIndex);
    EXPECT_EQ(4, issues.at(1).nodeIndex);
    EXPECT_EQ(GenomeIssueType::CycleAvoidingRootGene, issues.at(2).type);
    EXPECT_EQ(3, issues.at(2).geneIndex);
    EXPECT_EQ((std::vector{1, 3, 1}), issues.at(2).cycleGeneIndices);
    EXPECT_EQ(GenomeIssueType::GeneUnreachableFromRoot, issues.at(3).type);
    EXPECT_EQ(6, issues.at(3).geneIndex);
    EXPECT_EQ(GenomeIssueType::TooManyGenesWithSeparation, issues.at(4).type);
    EXPECT_EQ(1, issues.at(4).geneIndex);
    EXPECT_EQ(4, issues.at(4).nodeIndex);
    EXPECT_EQ((std::vector{0, 5}), issues.at(4).geneIndicesKeepingSeparation);
}

TEST_F(DescValidationServiceGenomeIssueTests, fix_removesGenesAndRemapsReferences)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1), createConstructorNode(3), createInjectorNode(2)}),
        GeneDesc().nodes({NodeDesc(), createVoidNode(), NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });

    auto newGeneIndexByOldGeneIndex = DescValidationService::get().fixGenomeIssues(genome);

    EXPECT_EQ((std::map<int, int>{{0, 0}, {3, 1}}), newGeneIndexByOldGeneIndex);
    auto expectedGenes = std::vector{
        GeneDesc().nodes({NodeDesc(), createConstructorNode(1), createInjectorNode(0)}),
        GeneDesc().nodes({NodeDesc()}),
    };
    EXPECT_EQ(expectedGenes, genome._genes);
    EXPECT_TRUE(findIssues(genome).empty());
}

TEST_F(DescValidationServiceGenomeIssueTests, fix_turnsOffConstructorAndSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({createConstructorNode(1, true), createConstructorNode(2, true), createConstructorNode(3, true)}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
        GeneDesc().nodes({createConstructorNode(3)}),
    });

    DescValidationService::get().fixGenomeIssues(genome);

    auto const& rootNodes = genome._genes.at(0)._nodes;
    EXPECT_TRUE(rootNodes.at(0)._constructor->_separation);
    EXPECT_TRUE(rootNodes.at(1)._constructor->_separation);
    EXPECT_FALSE(rootNodes.at(2)._constructor->_separation);
    EXPECT_EQ(std::nullopt, genome._genes.at(3)._nodes.at(0)._constructor);
    EXPECT_TRUE(findIssues(genome).empty());
}

TEST_F(DescValidationServiceGenomeIssueTests, fix_voidBoundaryNodeBecomesBaseCell)
{
    auto genome = GenomeDesc()
                      .genes({GeneDesc().nodes({NodeDesc(), NodeDesc().cellType(SensorGenomeDesc()), createVoidNode()})})
                      .mutationRates(MutationRatesDesc().deleteNodeMutation(DeleteNodeMutationDesc().nodeProbability(0.1f)));

    DescValidationService::get().fixGenomeIssues(genome);

    EXPECT_EQ(CellType_Base, genome._genes.at(0)._nodes.at(2).getCellType());
    EXPECT_TRUE(findIssues(genome).empty());
}

TEST_F(DescValidationServiceGenomeIssueTests, fix_removesConstructorOfCutOffVoidNode)
{
    auto genome = createTriangleWithVoidNodes3_8_12();
    genome._genes.at(0)._nodes.at(3).constructor(ConstructorGenomeDesc().geneIndex(0));

    DescValidationService::get().fixGenomeIssues(genome);

    EXPECT_EQ(std::nullopt, genome._genes.at(0)._nodes.at(3)._constructor);
    EXPECT_TRUE(findIssues(genome).empty());
}

TEST_F(DescValidationServiceGenomeIssueTests, fix_onlyGeneWithVoidNode_withNodeMutations_leavesNoIssues)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), createVoidNode(), NodeDesc()})}).mutationRates(createStructuralNodeMutationRates());

    DescValidationService::get().fixGenomeIssues(genome);

    EXPECT_TRUE(findIssues(genome).empty());
}

TEST_F(DescValidationServiceGenomeIssueTests, fix_randomGenomes_leaveNoIssues)
{
    std::mt19937 randomEngine(1);
    for (auto run : std::views::iota(0, 500)) {
        auto genome = DescTestDataFactory::get().createRandomGenome(randomEngine);
        if (run % 2 == 1) {
            genome._mutationRates = createStructuralNodeMutationRates();
        }

        DescValidationService::get().fixGenomeIssues(genome);

        EXPECT_TRUE(findIssues(genome).empty()) << "run " << run;
    }
}
