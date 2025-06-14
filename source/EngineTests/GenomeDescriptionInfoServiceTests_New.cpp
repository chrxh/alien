
#include <gtest/gtest.h>

#include "EngineInterface/GenomeDescriptionInfoService.h"
#include "EngineInterface/GenomeDescriptions.h"

class GenomeDescriptionInfoServiceTests_New : public ::testing::Test
{
public:
    virtual ~GenomeDescriptionInfoServiceTests_New() = default;
};

TEST_F(GenomeDescriptionInfoServiceTests_New, getNumberOfResultingCells_Empty)
{
    auto genome = GenomeDescription_New();
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(0, result);
}

TEST_F(GenomeDescriptionInfoServiceTests_New, getNumberOfResultingCells_oneReferencesOneSingleTimes)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
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

TEST_F(GenomeDescriptionInfoServiceTests_New, getNumberOfResultingCells_oneReferencesOneMultipleTimes)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
        }),
        GeneDescription().nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(12, result);
}

TEST_F(GenomeDescriptionInfoServiceTests_New, getNumberOfResultingCells_oneReferencesMany_depth1)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
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

TEST_F(GenomeDescriptionInfoServiceTests_New, getNumberOfResultingCells_oneReferencesMany_depth2)
{
    auto genome = GenomeDescription_New().genes({
        // Level 0
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
        }),
        // Level 1
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(3)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(4)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(5)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(6)),
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

TEST_F(GenomeDescriptionInfoServiceTests_New, getNumberOfResultingCells_manyReferenceOne)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
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

TEST_F(GenomeDescriptionInfoServiceTests_New, getNumberOfResultingCells_doNotCountUnreachable)
{
    auto genome = GenomeDescription_New().genes({
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

TEST_F(GenomeDescriptionInfoServiceTests_New, getNumberOfResultingCells_doNotCountPrincipalReferencesPrincipal)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(0)),
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(3, result);
}

TEST_F(GenomeDescriptionInfoServiceTests_New, getNumberOfResultingCells_doNotCountAuxiliaryReferencesPrincipal)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(0)),
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(5, result);
}

TEST_F(GenomeDescriptionInfoServiceTests_New, getNumberOfResultingCells_infinity_1cycle)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(-1, result);
}

TEST_F(GenomeDescriptionInfoServiceTests_New, getNumberOfResultingCells_infinity_2cycle)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription(),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(-1, result);
}

TEST_F(GenomeDescriptionInfoServiceTests_New, getNumberOfResultingCells_multipleBranchesAndConcatenations_withoutSeparation)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
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

TEST_F(GenomeDescriptionInfoServiceTests_New, getNumberOfResultingCells_multipleBranchesAndConcatenations_withSeparation)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription()
            .nodes({
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
            })
            .numBranches(std::nullopt)
            .numConcatenations(3),
    });
    auto result = GenomeDescriptionInfoService::get().getNumberOfResultingCells(genome);

    EXPECT_EQ(2 + 2 + 3 * 3 + 3 * 3, result);
}

TEST_F(GenomeDescriptionInfoServiceTests_New, getReferences)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
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

TEST_F(GenomeDescriptionInfoServiceTests_New, getReferencedBy)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription(),
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(0)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(0)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(0)),
        }),
    });
    auto result = GenomeDescriptionInfoService::get().getReferencedBy(genome, 0);

    ASSERT_EQ(3, result.size());
    EXPECT_EQ(1, result.at(0));
    EXPECT_EQ(1, result.at(1));
    EXPECT_EQ(2, result.at(2));
}
