
#include <gtest/gtest.h>
#include <algorithm>

#include "EngineInterface/GenomeDescriptionInfoService.h"
#include "EngineInterface/GenomeDescription.h"

class GenomeDescriptionInfoServiceTests : public ::testing::Test
{
public:
    GenomeDescriptionInfoServiceTests()
        : _genomeDescriptionInfoService(_genomeDescriptionInfoService) {
    }

    virtual ~GenomeDescriptionInfoServiceTests() = default;

protected:
    GenomeDescriptionInfoService const& _genomeDescriptionInfoService;
};

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_Empty)
{
    auto genome = GenomeDescription();
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(0, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_oneReferencesOneSingleTimes)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(6, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_oneReferencesOneMultipleTimes)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(12, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_oneReferencesMany_depth1)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(7, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_oneReferencesMany_depth2)
{
    auto genome = GenomeDescription().genes({
        // Level 0
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        // Level 1
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(3)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(4)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(5)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(6)),
            NodeDescription(),
        }),
        // Level 2
        GeneDescription().separation(true).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(2 + 2 + 3 + 2 + 2 + 3 + 1, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_manyReferenceOne)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(2 + 2 + 3 + 3, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_doNotCountUnreachable)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(1, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_doNotCountPrincipalReferencesPrincipal)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(3, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_doNotCountAuxiliaryReferencesPrincipal)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(5, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_infinity_1cycle)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(-1, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_infinity_2cycle)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(-1, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_multipleBranchesAndConcatenations_withoutSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription()
            .nodes({
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
            })
            .separation(false)
            .numBranches(2)
            .numConcatenations(3),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(2 + 2 + 3 * 2 * 3 + 3 * 2 * 3, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_multipleBranchesAndConcatenations_withSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription()
            .nodes({
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
            })
            .separation(true)
            .numConcatenations(3),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(2 + 2 + 3 * 3 + 3 * 3, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_multipleBranchesAndConcatenations_onFirstGene)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numBranches(10).numConcatenations(5).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(2 * 5 * 10, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_nestedMultipleBranches)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(false).numBranches(2).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).numBranches(2).nodes({
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getNumberOfResultingCells(genome);

    EXPECT_EQ(1 + 2 * (1 + 2 * 1), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getReferences)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getReferences(genome._genes.at(0));

    ASSERT_EQ(3, result.size());
    EXPECT_EQ(1, result.at(0));
    EXPECT_EQ(2, result.at(1));
    EXPECT_EQ(1, result.at(2));
}

TEST_F(GenomeDescriptionInfoServiceTests, getReferencedBy)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getReferencedBy(genome, 0);

    ASSERT_EQ(3, result.size());
    EXPECT_EQ(1, result.at(0));
    EXPECT_EQ(1, result.at(1));
    EXPECT_EQ(2, result.at(2));
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_empty)
{
    auto genome = GenomeDescription();
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_TRUE(result.empty());
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_singleNonSeparatingHull)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 2}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_twoNonSeparatingHulls)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(3)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(3)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(4)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(3)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(4)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 2}, {3, 4}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_threeNonSeparatingHulls)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(3)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(3)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(4)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(3)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(4)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 2}, {3}, {4}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_onlySeparatingGenes)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}, {2}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_disconnectedComponents)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(3)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1}, {2, 3}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_singleGeneGenome)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_genesWithoutNodes)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false),
        GeneDescription().separation(true),
        GeneDescription().separation(false),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}, {2}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_selfReferencingGene)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_referenceRootFromDifferentSubGenome)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_invalidGeneReferences)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(5)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(10)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(99)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_complexMixedSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(3)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(4)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(6)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(5)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 3}, {2}, {4, 6, 5}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_separatingGenesWithNonSeparatingReferences)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 2}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_largeGenomeWithManyReferences)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(3)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(4)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(5)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(6)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(7)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(8)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 2, 3, 5, 6, 7}, {4, 8}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_circularReferenceWithSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}, {2}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_circularReferenceWithoutSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 2}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_geneWithMultipleNodesAndDifferentCellTypes)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(BaseGenomeDescription()),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(BaseGenomeDescription()),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(BaseGenomeDescription()),
            NodeDescription().cellType(BaseGenomeDescription()),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_mixedReferencesNonConstructorAndConstructor)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription(),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(BaseGenomeDescription()),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_emptyNodesInGenes)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({}),
        GeneDescription().separation(true).nodes({}),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1}, {2, 0}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_deepNestedReferences)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(3)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(4)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0, 1, 2, 3, 4}}), result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getGeneIndicesForSubGenomes_alternatingPattern)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(3)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(4)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
    });
    auto result = _genomeDescriptionInfoService.getGeneIndicesForSubGenomes(genome);
    EXPECT_EQ((std::vector<std::vector<int>>{{0}, {1, 2}, {3, 4}}), result);
}