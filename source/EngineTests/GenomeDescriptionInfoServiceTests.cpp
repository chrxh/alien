
#include <gtest/gtest.h>
#include <algorithm>

#include "EngineInterface/GenomeDescriptionInfoService.h"
#include "EngineInterface/GenomeDescription.h"

class GenomeDescriptionInfoServiceTests : public ::testing::Test
{
public:
    virtual ~GenomeDescriptionInfoServiceTests() = default;
};

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_Empty)
{
    auto genome = GenomeDescription();
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(0, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_oneReferencesOneSingleTimes)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(6, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_oneReferencesOneMultipleTimes)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(12, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_oneReferencesMany_depth1)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
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
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(7, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_oneReferencesMany_depth2)
{
    auto genome = GenomeDescription().genes({
        // Level 0
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
        }),
        // Level 1
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(3)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(4)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(5)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(6)),
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
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(2 + 2 + 3 + 2 + 2 + 3 + 1, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_manyReferenceOne)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

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
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(1, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_doNotCountPrincipalReferencesPrincipal)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(3, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_doNotCountAuxiliaryReferencesPrincipal)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(5, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_infinity_1cycle)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(-1, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_infinity_2cycle)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(-1, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_multipleBranchesAndConcatenations_withoutSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
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
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(2 + 2 + 3 * 2 * 3 + 3 * 2 * 3, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_multipleBranchesAndConcatenations_withSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
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
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

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
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(2 * 5 * 10, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getReferences)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getReferences(genome._genes.at(0));

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
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getReferencedBy(genome, 0);

    ASSERT_EQ(3, result.size());
    EXPECT_EQ(1, result.at(0));
    EXPECT_EQ(1, result.at(1));
    EXPECT_EQ(2, result.at(2));
}

TEST_F(GenomeDescriptionInfoServiceTests, getReferencedGenesInNonSeparatingGeneHull_onlyNonSeparatingGenes)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getReferencedGenesInNonSeparatingGeneHull(genome, 0);

    // All genes should be in non-separating list
    ASSERT_EQ(3, result.nonSeparatingGeneIndices.size());
    EXPECT_EQ(0, result.separatingGeneIndices.size());
    
    // Check if genes 0, 1, 2 are all included (order may vary)
    std::sort(result.nonSeparatingGeneIndices.begin(), result.nonSeparatingGeneIndices.end());
    EXPECT_EQ(0, result.nonSeparatingGeneIndices[0]);
    EXPECT_EQ(1, result.nonSeparatingGeneIndices[1]);
    EXPECT_EQ(2, result.nonSeparatingGeneIndices[2]);
}

TEST_F(GenomeDescriptionInfoServiceTests, getReferencedGenesInNonSeparatingGeneHull_mixedSeparatingAndNonSeparating)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getReferencedGenesInNonSeparatingGeneHull(genome, 0);

    // Should have genes 0,1 as non-separating and gene 2 as separating
    ASSERT_EQ(2, result.nonSeparatingGeneIndices.size());
    ASSERT_EQ(1, result.separatingGeneIndices.size());
    
    std::sort(result.nonSeparatingGeneIndices.begin(), result.nonSeparatingGeneIndices.end());
    EXPECT_EQ(0, result.nonSeparatingGeneIndices[0]);
    EXPECT_EQ(1, result.nonSeparatingGeneIndices[1]);
    EXPECT_EQ(2, result.separatingGeneIndices[0]);
}

TEST_F(GenomeDescriptionInfoServiceTests, getReferencedGenesInNonSeparatingGeneHull_startFromDifferentIndex)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getReferencedGenesInNonSeparatingGeneHull(genome, 1);

    // Starting from gene 1, should reach genes 1 and 2 (not 0)
    ASSERT_EQ(2, result.nonSeparatingGeneIndices.size());
    EXPECT_EQ(0, result.separatingGeneIndices.size());
    
    std::sort(result.nonSeparatingGeneIndices.begin(), result.nonSeparatingGeneIndices.end());
    EXPECT_EQ(1, result.nonSeparatingGeneIndices[0]);
    EXPECT_EQ(2, result.nonSeparatingGeneIndices[1]);
}

TEST_F(GenomeDescriptionInfoServiceTests, getReferencedGenesInNonSeparatingGeneHull_separatingGeneBlocksTraversal)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getReferencedGenesInNonSeparatingGeneHull(genome, 0);

    // Should only reach gene 0 (non-separating) and gene 1 (separating, but not traversed)
    // Gene 2 should not be reached because we don't traverse through separating gene 1
    ASSERT_EQ(1, result.nonSeparatingGeneIndices.size());
    ASSERT_EQ(1, result.separatingGeneIndices.size());
    
    EXPECT_EQ(0, result.nonSeparatingGeneIndices[0]);
    EXPECT_EQ(1, result.separatingGeneIndices[0]);
}
