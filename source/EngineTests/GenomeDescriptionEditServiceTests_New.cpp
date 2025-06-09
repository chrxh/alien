
#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/GenomeDescriptionInfoService.h"
#include "EngineInterface/GenomeDescriptions.h"

#include <gtest/gtest.h>

class GenomeDescriptionEditServiceTests_New : public ::testing::Test
{
public:
    virtual ~GenomeDescriptionEditServiceTests_New() = default;
};

TEST_F(GenomeDescriptionEditServiceTests_New, addEmptyGene_onEmptyGenome)
{
    auto genome = GenomeDescription_New();
    GenomeDescriptionEditService::get().addEmptyGene(genome, 0);

    EXPECT_EQ(1, genome._genes.size());
}

TEST_F(GenomeDescriptionEditServiceTests_New, addEmptyGene_onNonEmptyGenomeAtBeginning)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription(),
        }),
    });
    GenomeDescriptionEditService::get().addEmptyGene(genome, 0);

    EXPECT_EQ(4, genome._genes.size());
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(i == 1 ? 0 : 1, genome._genes.at(i)._nodes.size());
    }
}

TEST_F(GenomeDescriptionEditServiceTests_New, addEmptyGene_onNonEmptyGenomeAtEnd)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription(),
        }),
    });
    GenomeDescriptionEditService::get().addEmptyGene(genome, 2);

    ASSERT_EQ(4, genome._genes.size());
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(i == 3 ? 0 : 1, genome._genes.at(i)._nodes.size());
    }
}

TEST_F(GenomeDescriptionEditServiceTests_New, addEmptyGene_withReferences)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(0)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(0)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(0)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
        }),
    });
    GenomeDescriptionEditService::get().addEmptyGene(genome, 1);

    ASSERT_EQ(4, genome._genes.size());
    for (int i = 0; i < 4; ++i) {
        auto const& gene = genome._genes.at(i);
        ASSERT_EQ(i == 2 ? 0 : 3, gene._nodes.size());
        if (i != 2) {
            EXPECT_EQ(0, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(0)._cellTypeData)._constructGeneIndex);
            EXPECT_EQ(1, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(1)._cellTypeData)._constructGeneIndex);
            EXPECT_EQ(3, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(2)._cellTypeData)._constructGeneIndex);
        }
    }
}

TEST_F(GenomeDescriptionEditServiceTests_New, removeGene_onNonEmptyGenomeAtMiddle_withReferences)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(0)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(0)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
             NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(0)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
             NodeDescription(),
             NodeDescription(),
        }),
    });
    GenomeDescriptionEditService::get().removeGene(genome, 1);

    ASSERT_EQ(2, genome._genes.size());
    EXPECT_EQ(3, genome._genes.at(0)._nodes.size());
    EXPECT_EQ(5, genome._genes.at(1)._nodes.size());
    for (int i = 0; i < 2; ++i) {
        auto const& gene = genome._genes.at(i);
        EXPECT_EQ(0, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(0)._cellTypeData)._constructGeneIndex);
        EXPECT_EQ(0, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(1)._cellTypeData)._constructGeneIndex);
        EXPECT_EQ(1, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(2)._cellTypeData)._constructGeneIndex);
    }
}

TEST_F(GenomeDescriptionEditServiceTests_New, removeGene_onNonEmptyGenomeAtEnd_withReferences)
{
    auto genome = GenomeDescription_New().genes({
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(0)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(0)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription().nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(0)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription_New().constructGeneIndex(2)),
            NodeDescription(),
            NodeDescription(),
        }),
    });
    GenomeDescriptionEditService::get().removeGene(genome, 2);

    ASSERT_EQ(2, genome._genes.size());
    EXPECT_EQ(3, genome._genes.at(0)._nodes.size());
    EXPECT_EQ(4, genome._genes.at(1)._nodes.size());
    for (int i = 0; i < 2; ++i) {
        auto const& gene = genome._genes.at(i);
        EXPECT_EQ(0, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(0)._cellTypeData)._constructGeneIndex);
        EXPECT_EQ(1, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(1)._cellTypeData)._constructGeneIndex);
        EXPECT_EQ(1, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(2)._cellTypeData)._constructGeneIndex);
    }
}
