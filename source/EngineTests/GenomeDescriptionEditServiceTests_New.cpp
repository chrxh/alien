
#include <gtest/gtest.h>

#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/GenomeDescriptionInfoService.h"
#include "EngineInterface/GenomeDescriptions.h"

class GenomeDescriptionEditServiceTests_New : public ::testing::Test
{
public:
    virtual ~GenomeDescriptionEditServiceTests_New() = default;

protected:
    GenomeDescription_New createGenome_3genes_3_4_5nodes()
    {
        return GenomeDescription_New().genes({
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
    }
};

TEST_F(GenomeDescriptionEditServiceTests_New, addEmptyGene_onEmptyGenome)
{
    auto genome = GenomeDescription_New();
    GenomeDescriptionEditService::get().addEmptyGene(genome, 0);

    EXPECT_EQ(1, genome._genes.size());
}

TEST_F(GenomeDescriptionEditServiceTests_New, addEmptyGene_onNonEmptyGenome_start)
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

TEST_F(GenomeDescriptionEditServiceTests_New, addEmptyGene_onNonEmptyGenome_end)
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
    auto genome = createGenome_3genes_3_4_5nodes();
    GenomeDescriptionEditService::get().addEmptyGene(genome, 1);

    ASSERT_EQ(4, genome._genes.size());
    for (int i = 0; i < 4; ++i) {
        auto const& gene = genome._genes.at(i);
        if (i == 0) {
            ASSERT_EQ(3, gene._nodes.size());
        } else if (i == 1) {
            ASSERT_EQ(4, gene._nodes.size());
        } else if (i == 2) {
            ASSERT_EQ(0, gene._nodes.size());
        } else if (i == 3) {
            ASSERT_EQ(5, gene._nodes.size());
        }
        if (i != 2) {
            EXPECT_EQ(0, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(0)._cellTypeData)._constructGeneIndex);
            EXPECT_EQ(1, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(1)._cellTypeData)._constructGeneIndex);
            EXPECT_EQ(3, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(2)._cellTypeData)._constructGeneIndex);
        }
    }
}

TEST_F(GenomeDescriptionEditServiceTests_New, removeGene_middle)
{
    auto genome = createGenome_3genes_3_4_5nodes();
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

TEST_F(GenomeDescriptionEditServiceTests_New, removeGene_end)
{
    auto genome = createGenome_3genes_3_4_5nodes();
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

TEST_F(GenomeDescriptionEditServiceTests_New, swapGenes)
{
    auto genome = createGenome_3genes_3_4_5nodes();
    GenomeDescriptionEditService::get().swapGenes(genome, 1);

    ASSERT_EQ(3, genome._genes.size());
    EXPECT_EQ(3, genome._genes.at(0)._nodes.size());
    EXPECT_EQ(5, genome._genes.at(1)._nodes.size());
    EXPECT_EQ(4, genome._genes.at(2)._nodes.size());
    for (int i = 0; i < 3; ++i) {
        auto const& gene = genome._genes.at(i);
        EXPECT_EQ(0, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(0)._cellTypeData)._constructGeneIndex);
        EXPECT_EQ(2, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(1)._cellTypeData)._constructGeneIndex);
        EXPECT_EQ(1, std::get<ConstructorGenomeDescription_New>(gene._nodes.at(2)._cellTypeData)._constructGeneIndex);
    }
}

TEST_F(GenomeDescriptionEditServiceTests_New, addEmptyNode_start)
{
    auto gene = GeneDescription().nodes({
        NodeDescription().cellTypeData(DepotGenomeDescription()),
        NodeDescription().cellTypeData(ConstructorGenomeDescription_New()),
        NodeDescription().cellTypeData(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().addEmptyNode(gene, 0);

    ASSERT_EQ(4, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Base, gene._nodes.at(1).getCellType());
    EXPECT_EQ(CellTypeGenome_Constructor, gene._nodes.at(2).getCellType());
    EXPECT_EQ(CellTypeGenome_Sensor, gene._nodes.at(3).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests_New, addEmptyNode_middle)
{
    auto gene = GeneDescription().nodes({
        NodeDescription().cellTypeData(DepotGenomeDescription()),
        NodeDescription().cellTypeData(ConstructorGenomeDescription_New()),
        NodeDescription().cellTypeData(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().addEmptyNode(gene, 1);

    ASSERT_EQ(4, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Constructor, gene._nodes.at(1).getCellType());
    EXPECT_EQ(CellTypeGenome_Base, gene._nodes.at(2).getCellType());
    EXPECT_EQ(CellTypeGenome_Sensor, gene._nodes.at(3).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests_New, addEmptyNode_end)
{
    auto gene = GeneDescription().nodes({
        NodeDescription().cellTypeData(DepotGenomeDescription()),
        NodeDescription().cellTypeData(ConstructorGenomeDescription_New()),
        NodeDescription().cellTypeData(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().addEmptyNode(gene, 2);

    ASSERT_EQ(4, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Constructor, gene._nodes.at(1).getCellType());
    EXPECT_EQ(CellTypeGenome_Sensor, gene._nodes.at(2).getCellType());
    EXPECT_EQ(CellTypeGenome_Base, gene._nodes.at(3).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests_New, removeNode_start)
{
    auto gene = GeneDescription().nodes({
        NodeDescription().cellTypeData(DepotGenomeDescription()),
        NodeDescription().cellTypeData(ConstructorGenomeDescription_New()),
        NodeDescription().cellTypeData(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().removeNode(gene, 0);

    ASSERT_EQ(2, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Constructor, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Sensor, gene._nodes.at(1).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests_New, removeNode_middle)
{
    auto gene = GeneDescription().nodes({
        NodeDescription().cellTypeData(DepotGenomeDescription()),
        NodeDescription().cellTypeData(ConstructorGenomeDescription_New()),
        NodeDescription().cellTypeData(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().removeNode(gene, 1);

    ASSERT_EQ(2, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Sensor, gene._nodes.at(1).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests_New, removeNode_end)
{
    auto gene = GeneDescription().nodes({
        NodeDescription().cellTypeData(DepotGenomeDescription()),
        NodeDescription().cellTypeData(ConstructorGenomeDescription_New()),
        NodeDescription().cellTypeData(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().removeNode(gene, 2);

    ASSERT_EQ(2, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Constructor, gene._nodes.at(1).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests_New, swapNodes)
{
    auto gene = GeneDescription().nodes({
        NodeDescription().cellTypeData(DepotGenomeDescription()),
        NodeDescription().cellTypeData(ConstructorGenomeDescription_New()),
        NodeDescription().cellTypeData(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().swapNodes(gene, 1);

    ASSERT_EQ(3, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Sensor, gene._nodes.at(1).getCellType());
    EXPECT_EQ(CellTypeGenome_Constructor, gene._nodes.at(2).getCellType());
}
