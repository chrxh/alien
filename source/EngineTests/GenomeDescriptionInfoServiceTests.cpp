
#include <gtest/gtest.h>

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
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().nodes({
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
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().nodes({
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
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().nodes({
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().nodes({
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
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
        }),
        // Level 1
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(3)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(4)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(5)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(6)),
            NodeDescription(),
        }),
        // Level 2
        GeneDescription().nodes({
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(2 + 2 + 3 + 2 + 2 + 3 + 1, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_manyReferenceOne)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription().nodes({
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
        GeneDescription().nodes({
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(1, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_doNotCountPrincipalReferencesPrincipal)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({
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
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().nodes({
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
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().nodes({
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
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription().nodes({
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
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription()
            .nodes({
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
            })
            .numBranches(2)
            .numConcatenations(3),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(2 + 2 + 3 * 2 * 3 + 3 * 2 * 3, result);
}

TEST_F(GenomeDescriptionInfoServiceTests, getNumberOfResultingCells_multipleBranchesAndConcatenations_withSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().nodes({
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
        GeneDescription().numBranches(10).numConcatenations(5).nodes({
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
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
        }),
        GeneDescription().nodes({
            NodeDescription(),
        }),
        GeneDescription().nodes({
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
        GeneDescription().nodes({
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getReferencedBy(genome, 0);

    ASSERT_EQ(3, result.size());
    EXPECT_EQ(1, result.at(0));
    EXPECT_EQ(1, result.at(1));
    EXPECT_EQ(2, result.at(2));
}
