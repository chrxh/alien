
#include <algorithm>

#include <gtest/gtest.h>

#include <EngineInterface/GenomeDesc.h>
#include <EngineInterface/GenomeDescInfoService.h>

class GenomeDescInfoServiceTests : public ::testing::Test
{
public:
    GenomeDescInfoServiceTests()
        : _genomeDescriptionInfoService(_genomeDescriptionInfoService)
    {}

    virtual ~GenomeDescInfoServiceTests() = default;

protected:
    GenomeDescInfoService const& _genomeDescriptionInfoService;
};

TEST_F(GenomeDescInfoServiceTests, getNumberOfResultingCells_Empty)
{
    auto genome = GenomeDesc();
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(0, result);
}

TEST_F(GenomeDescInfoServiceTests, getNumberOfResultingCells_oneReferencesOneSingleTimes)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
            NodeDesc(),
            NodeDesc(),
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(6, result);
}

TEST_F(GenomeDescInfoServiceTests, getNumberOfResultingCells_oneReferencesOneMultipleTimes)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(12, result);
}

TEST_F(GenomeDescInfoServiceTests, getNumberOfResultingCells_oneReferencesMany_depth1)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc(),
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(7, result);
}

TEST_F(GenomeDescInfoServiceTests, getNumberOfResultingCells_oneReferencesMany_depth2)
{
    auto genome = GenomeDesc().genes({
        // Level 0
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(true)),
        }),
        // Level 1
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(4).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(5).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(6).separation(true)),
            NodeDesc(),
        }),
        // Level 2
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc(),
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(2 + 2 + 3 + 2 + 2 + 3 + 1, result);
}

TEST_F(GenomeDescInfoServiceTests, getNumberOfResultingCells_manyReferenceOne)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(true)),
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc(),
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(2 + 2 + 3 + 3, result);
}

TEST_F(GenomeDescInfoServiceTests, getNumberOfResultingCells_doNotCountUnreachable)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(1, result);
}

TEST_F(GenomeDescInfoServiceTests, getNumberOfResultingCells_doNotCountPrincipalReferencesPrincipal)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(true)),
            NodeDesc(),
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(3, result);
}

TEST_F(GenomeDescInfoServiceTests, getNumberOfResultingCells_doNotCountAuxiliaryReferencesPrincipal)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
            NodeDesc(),
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(true)),
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(5, result);
}

TEST_F(GenomeDescInfoServiceTests, getNumberOfResultingCells_infinity_1cycle)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
            NodeDesc(),
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(-1, result);
}

TEST_F(GenomeDescInfoServiceTests, getNumberOfResultingCells_infinity_2cycle)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
            NodeDesc(),
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(true)),
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(-1, result);
}

TEST_F(GenomeDescInfoServiceTests, getNumberOfResultingCells_multipleBranchesAndConcatenations_withoutSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false).numBranches(2).numConcatenations(3)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false).numBranches(2).numConcatenations(3)),
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc(),
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(2 + 2 + 3 * 2 * 3 + 3 * 2 * 3, result);
}

TEST_F(GenomeDescInfoServiceTests, getNumberOfResultingCells_multipleBranchesAndConcatenations_withSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(true).numConcatenations(3)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(true).numConcatenations(3)),
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc(),
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(2 + 2 + 3 * 3 + 3 * 3, result);
}

TEST_F(GenomeDescInfoServiceTests, getNumberOfResultingCells_nestedMultipleBranches)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false).numBranches(2)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false).numBranches(2)),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(1 + 2 * (1 + 2 * 1), result);
}

TEST_F(GenomeDescInfoServiceTests, getReferences)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getReferences(genome._genes.at(0));

    ASSERT_EQ(3, result.size());
    EXPECT_EQ(1, result.at(0));
    EXPECT_EQ(2, result.at(1));
    EXPECT_EQ(1, result.at(2));
}

TEST_F(GenomeDescInfoServiceTests, getReferencedBy)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(true)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getReferencedBy(genome, 0);

    ASSERT_EQ(3, result.size());
    EXPECT_EQ(1, result.at(0));
    EXPECT_EQ(1, result.at(1));
    EXPECT_EQ(2, result.at(2));
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_empty)
{
    auto genome = GenomeDesc();
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_TRUE(result.empty());
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_singleNonSeparatingHull)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 2}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_twoNonSeparatingHulls)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(4).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(4).separation(false)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 2}, {3, 4}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_threeNonSeparatingHulls)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(4).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3).separation(true)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(4).separation(true)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 2}, {3}, {4}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_onlySeparatingGenes)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}, {2}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_disconnectedComponents)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1}, {2, 3}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_singleGeneGenome)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_genesWithoutNodes)
{
    auto genome = GenomeDesc().genes({
        GeneDesc(),
        GeneDesc(),
        GeneDesc(),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}, {2}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_selfReferencingGene)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(false)),
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_referenceRootFromDifferentSubGenome)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(true)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_invalidGeneReferences)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(5)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(10)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(99)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_complexMixedSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(4).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(6).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(5).separation(false)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 3}, {2}, {4, 6, 5}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_nonSeparatingReferences)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 2}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_largeGenomeWithManyReferences)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(4).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(5).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(6).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(7).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(8).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 2, 3, 5, 6, 7}, {4, 8}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_circularReferenceWithSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(true)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}, {2}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_circularReferenceWithoutSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(false)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 2}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_geneWithMultipleNodesAndDifferentCellTypes)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().cellType(BaseGenomeDesc()),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc().cellType(BaseGenomeDesc()),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().cellType(BaseGenomeDesc()),
            NodeDesc().cellType(BaseGenomeDesc()),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_mixedReferencesNonConstructorAndConstructor)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc(),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc().cellType(BaseGenomeDesc()),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_emptyNodesInGenes)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({}),
        GeneDesc().nodes({}),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(false)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}, {2, 0}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_deepNestedReferences)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(4).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 2, 3, 4}}), result);
}

TEST_F(GenomeDescInfoServiceTests, getGeneIndicesForSubGenomes_alternatingPattern)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(3).separation(true)),
        }),
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(4).separation(false)),
        }),
        GeneDesc().nodes({
            NodeDesc(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1, 2}, {3, 4}}), result);
}
