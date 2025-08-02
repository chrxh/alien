
#include <gtest/gtest.h>

#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/GenomeDescriptionInfoService.h"
#include "EngineInterface/GenomeDescription.h"

class GenomeDescriptionEditServiceTests : public ::testing::Test
{
public:
    virtual ~GenomeDescriptionEditServiceTests() = default;

protected:
    int getRefGeneIndex(GeneDescription const& gene, int nodeIndex) const
    {
        return std::get<ConstructorGenomeDescription>(gene._nodes.at(nodeIndex)._cellTypeData)._geneIndex;
    };

    GenomeDescription createGenome_complexCycles() const
    {
        return GenomeDescription().genes({
            GeneDescription().separation(true).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
            }),
            GeneDescription().separation(true).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
                NodeDescription(),
            }),
            GeneDescription().separation(true).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
                NodeDescription(),
                NodeDescription(),
            }),
        });
    }

    GenomeDescription createGenome_subCycle() const
    {
        return GenomeDescription().genes({
            GeneDescription().separation(true).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription(),
            }),
            GeneDescription().separation(true).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
                NodeDescription(),
            }),
            GeneDescription().separation(true).nodes({
                NodeDescription(),
                NodeDescription(),
            }),
        });
    }

    GenomeDescription createGenome_noCycles() const
    {
        return GenomeDescription().genes({
            GeneDescription().separation(true).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription(),
            }),
            GeneDescription().separation(true).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
                NodeDescription(),
            }),
            GeneDescription().separation(true).nodes({
                NodeDescription(),
                NodeDescription(),
            }),
        });
    }

};

TEST_F(GenomeDescriptionEditServiceTests, addEmptyGene_onEmptyGenome)
{
    auto genome = GenomeDescription();
    GenomeDescriptionEditService::get().addGene(genome, 0, GeneDescription().separation(true));

    EXPECT_EQ(1, genome._genes.size());
}

TEST_F(GenomeDescriptionEditServiceTests, addEmptyGene_onNonEmptyGenome_start)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
    });
    GenomeDescriptionEditService::get().addGene(genome, 0, GeneDescription().separation(true));

    EXPECT_EQ(4, genome._genes.size());
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(i == 1 ? 0 : 1, genome._genes.at(i)._nodes.size());
    }
}

TEST_F(GenomeDescriptionEditServiceTests, addEmptyGene_onNonEmptyGenome_end)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
    });
    GenomeDescriptionEditService::get().addGene(genome, 2, GeneDescription().separation(true));

    ASSERT_EQ(4, genome._genes.size());
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(i == 3 ? 0 : 1, genome._genes.at(i)._nodes.size());
    }
}

TEST_F(GenomeDescriptionEditServiceTests, addEmptyGene_withReferences)
{
    auto genome = createGenome_complexCycles();
    GenomeDescriptionEditService::get().addGene(genome, 1, GeneDescription().separation(true));

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
            EXPECT_EQ(0, std::get<ConstructorGenomeDescription>(gene._nodes.at(0)._cellTypeData)._geneIndex);
            EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(gene._nodes.at(1)._cellTypeData)._geneIndex);
            EXPECT_EQ(3, std::get<ConstructorGenomeDescription>(gene._nodes.at(2)._cellTypeData)._geneIndex);
        }
    }
}

TEST_F(GenomeDescriptionEditServiceTests, removeGene_middle)
{
    auto genome = createGenome_complexCycles();
    GenomeDescriptionEditService::get().removeGene(genome, 1);

    ASSERT_EQ(2, genome._genes.size());
    EXPECT_EQ(3, genome._genes.at(0)._nodes.size());
    EXPECT_EQ(5, genome._genes.at(1)._nodes.size());
    for (int i = 0; i < 2; ++i) {
        auto const& gene = genome._genes.at(i);
        EXPECT_EQ(0, std::get<ConstructorGenomeDescription>(gene._nodes.at(0)._cellTypeData)._geneIndex);
        EXPECT_EQ(0, std::get<ConstructorGenomeDescription>(gene._nodes.at(1)._cellTypeData)._geneIndex);
        EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(gene._nodes.at(2)._cellTypeData)._geneIndex);
    }
}

TEST_F(GenomeDescriptionEditServiceTests, removeGene_end)
{
    auto genome = createGenome_complexCycles();
    GenomeDescriptionEditService::get().removeGene(genome, 2);

    ASSERT_EQ(2, genome._genes.size());
    EXPECT_EQ(3, genome._genes.at(0)._nodes.size());
    EXPECT_EQ(4, genome._genes.at(1)._nodes.size());
    for (int i = 0; i < 2; ++i) {
        auto const& gene = genome._genes.at(i);
        EXPECT_EQ(0, std::get<ConstructorGenomeDescription>(gene._nodes.at(0)._cellTypeData)._geneIndex);
        EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(gene._nodes.at(1)._cellTypeData)._geneIndex);
        EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(gene._nodes.at(2)._cellTypeData)._geneIndex);
    }
}

TEST_F(GenomeDescriptionEditServiceTests, swapGenes)
{
    auto genome = createGenome_complexCycles();
    GenomeDescriptionEditService::get().swapGenes(genome, 1);

    ASSERT_EQ(3, genome._genes.size());
    EXPECT_EQ(3, genome._genes.at(0)._nodes.size());
    EXPECT_EQ(5, genome._genes.at(1)._nodes.size());
    EXPECT_EQ(4, genome._genes.at(2)._nodes.size());
    for (int i = 0; i < 3; ++i) {
        auto const& gene = genome._genes.at(i);
        EXPECT_EQ(0, std::get<ConstructorGenomeDescription>(gene._nodes.at(0)._cellTypeData)._geneIndex);
        EXPECT_EQ(2, std::get<ConstructorGenomeDescription>(gene._nodes.at(1)._cellTypeData)._geneIndex);
        EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(gene._nodes.at(2)._cellTypeData)._geneIndex);
    }
}

TEST_F(GenomeDescriptionEditServiceTests, addEmptyNode_start)
{
    auto gene = GeneDescription().separation(true).nodes({
        NodeDescription().cellTypeData(DepotGenomeDescription()),
        NodeDescription().cellTypeData(ConstructorGenomeDescription()),
        NodeDescription().cellTypeData(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().addNode(gene, 0, NodeDescription());

    ASSERT_EQ(4, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Base, gene._nodes.at(1).getCellType());
    EXPECT_EQ(CellTypeGenome_Constructor, gene._nodes.at(2).getCellType());
    EXPECT_EQ(CellTypeGenome_Sensor, gene._nodes.at(3).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests, addEmptyNode_middle)
{
    auto gene = GeneDescription().separation(true).nodes({
        NodeDescription().cellTypeData(DepotGenomeDescription()),
        NodeDescription().cellTypeData(ConstructorGenomeDescription()),
        NodeDescription().cellTypeData(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().addNode(gene, 1, NodeDescription());

    ASSERT_EQ(4, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Constructor, gene._nodes.at(1).getCellType());
    EXPECT_EQ(CellTypeGenome_Base, gene._nodes.at(2).getCellType());
    EXPECT_EQ(CellTypeGenome_Sensor, gene._nodes.at(3).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests, addEmptyNode_end)
{
    auto gene = GeneDescription().separation(true).nodes({
        NodeDescription().cellTypeData(DepotGenomeDescription()),
        NodeDescription().cellTypeData(ConstructorGenomeDescription()),
        NodeDescription().cellTypeData(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().addNode(gene, 2, NodeDescription());

    ASSERT_EQ(4, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Constructor, gene._nodes.at(1).getCellType());
    EXPECT_EQ(CellTypeGenome_Sensor, gene._nodes.at(2).getCellType());
    EXPECT_EQ(CellTypeGenome_Base, gene._nodes.at(3).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests, removeNode_start)
{
    auto gene = GeneDescription().separation(true).nodes({
        NodeDescription().cellTypeData(DepotGenomeDescription()),
        NodeDescription().cellTypeData(ConstructorGenomeDescription()),
        NodeDescription().cellTypeData(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().removeNode(gene, 0);

    ASSERT_EQ(2, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Constructor, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Sensor, gene._nodes.at(1).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests, removeNode_middle)
{
    auto gene = GeneDescription().separation(true).nodes({
        NodeDescription().cellTypeData(DepotGenomeDescription()),
        NodeDescription().cellTypeData(ConstructorGenomeDescription()),
        NodeDescription().cellTypeData(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().removeNode(gene, 1);

    ASSERT_EQ(2, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Sensor, gene._nodes.at(1).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests, removeNode_end)
{
    auto gene = GeneDescription().separation(true).nodes({
        NodeDescription().cellTypeData(DepotGenomeDescription()),
        NodeDescription().cellTypeData(ConstructorGenomeDescription()),
        NodeDescription().cellTypeData(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().removeNode(gene, 2);

    ASSERT_EQ(2, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Constructor, gene._nodes.at(1).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests, swapNodes)
{
    auto gene = GeneDescription().separation(true).nodes({
        NodeDescription().cellTypeData(DepotGenomeDescription()),
        NodeDescription().cellTypeData(ConstructorGenomeDescription()),
        NodeDescription().cellTypeData(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().swapNodes(gene, 1);

    ASSERT_EQ(3, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Sensor, gene._nodes.at(1).getCellType());
    EXPECT_EQ(CellTypeGenome_Constructor, gene._nodes.at(2).getCellType());
}

class GenomeDescriptionEditServiceTests_AllStartGenes
    : public GenomeDescriptionEditServiceTests
    , public testing::WithParamInterface<int>
{};

INSTANTIATE_TEST_SUITE_P(GenomeDescriptionEditServiceTests_AllStartGenes, GenomeDescriptionEditServiceTests_AllStartGenes, ::testing::Values(0, 1, 2));

TEST_P(GenomeDescriptionEditServiceTests_AllStartGenes, adaptDescriptionForPreview_complexCycles)
{
    auto startGeneIndex = GetParam();
    auto genome = createGenome_complexCycles();
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(genome, startGeneIndex);

    ASSERT_EQ(3, genome._genes.size());

    auto const& gene0 = genome._genes.at(0);
    ASSERT_EQ(3, gene0._nodes.size());
    EXPECT_EQ(3, getRefGeneIndex(gene0, 0));
    EXPECT_EQ(startGeneIndex == 0 ? 1 : 3, getRefGeneIndex(gene0, 1));
    EXPECT_EQ(startGeneIndex == 0 ? 2 : 3, getRefGeneIndex(gene0, 2));

    auto const& gene1 = genome._genes.at(1);
    ASSERT_EQ(4, gene1._nodes.size());
    EXPECT_EQ(startGeneIndex == 1 ? 0 : 3, getRefGeneIndex(gene1, 0));
    EXPECT_EQ(3, getRefGeneIndex(gene1, 1));
    EXPECT_EQ(startGeneIndex == 1 ? 2 : 3, getRefGeneIndex(gene1, 2));

    auto const& gene2 = genome._genes.at(2);
    ASSERT_EQ(5, gene2._nodes.size());
    EXPECT_EQ(startGeneIndex == 2 ? 0 : 3, getRefGeneIndex(gene2, 0));
    EXPECT_EQ(startGeneIndex == 2 ? 1 : 3, getRefGeneIndex(gene2, 1));
    EXPECT_EQ(3, getRefGeneIndex(gene2, 2));
}

TEST_P(GenomeDescriptionEditServiceTests_AllStartGenes, adaptDescriptionForPreview_subCycle)
{
    auto startGeneIndex = GetParam();
    auto genome = createGenome_subCycle();
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(genome, startGeneIndex);

    ASSERT_EQ(3, genome._genes.size());

    auto const& gene0 = genome._genes.at(0);
    ASSERT_EQ(2, gene0._nodes.size());
    EXPECT_EQ(startGeneIndex == 1 ? 3 : 1, getRefGeneIndex(gene0, 0));

    auto const& gene1 = genome._genes.at(1);
    ASSERT_EQ(2, gene1._nodes.size());
    EXPECT_EQ(startGeneIndex == 0 ? 3 : 0, getRefGeneIndex(gene1, 0));

    auto const& gene2 = genome._genes.at(2);
    ASSERT_EQ(2, gene2._nodes.size());
}

TEST_P(GenomeDescriptionEditServiceTests_AllStartGenes, adaptDescriptionForPreview_noCycles)
{
    auto startGeneIndex = GetParam();
    auto genome = createGenome_noCycles();
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(genome, startGeneIndex);

    ASSERT_EQ(3, genome._genes.size());

    auto const& gene0 = genome._genes.at(0);
    ASSERT_EQ(2, gene0._nodes.size());
    EXPECT_EQ(1, getRefGeneIndex(gene0, 0));

    auto const& gene1 = genome._genes.at(1);
    ASSERT_EQ(2, gene1._nodes.size());
    EXPECT_EQ(2, getRefGeneIndex(gene1, 0));

    auto const& gene2 = genome._genes.at(2);
    ASSERT_EQ(2, gene2._nodes.size());
}

TEST_F(GenomeDescriptionEditServiceTests, adaptDescriptionForPreview_emptyGenome)
{
    auto genome = GenomeDescription();
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(genome, 0);
    
    EXPECT_EQ(0, genome._genes.size());
}

TEST_F(GenomeDescriptionEditServiceTests, adaptDescriptionForPreview_invalidGeneReference)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(5)),  // Invalid reference beyond genome size
            NodeDescription(),
        }),
    });
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(genome, 0);

    ASSERT_EQ(1, genome._genes.size());
    ASSERT_EQ(2, genome._genes.at(0)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(0)._nodes.at(0).getCellType());
    // Invalid reference should remain unchanged (the castrate method has bounds checking)
    EXPECT_EQ(5, std::get<ConstructorGenomeDescription>(genome._genes.at(0)._nodes.at(0)._cellTypeData)._geneIndex);
}