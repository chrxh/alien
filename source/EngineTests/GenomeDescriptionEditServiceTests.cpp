
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
            GeneDescription().separation(false).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
            }),
            GeneDescription().separation(false).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
                NodeDescription(),
            }),
            GeneDescription().separation(false).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
                NodeDescription(),
                NodeDescription(),
            }),
        });
    }
};

TEST_F(GenomeDescriptionEditServiceTests, addEmptyGene_onEmptyGenome)
{
    auto genome = GenomeDescription();
    GenomeDescriptionEditService::get().addGene(genome, 0, GeneDescription().separation(false));

    EXPECT_EQ(1, genome._genes.size());
}

TEST_F(GenomeDescriptionEditServiceTests, addEmptyGene_onNonEmptyGenome_start)
{
    auto genome = GenomeDescription().genes({
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
    GenomeDescriptionEditService::get().addGene(genome, 0, GeneDescription().separation(false));

    EXPECT_EQ(4, genome._genes.size());
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(i == 1 ? 0 : 1, genome._genes.at(i)._nodes.size());
    }
}

TEST_F(GenomeDescriptionEditServiceTests, addEmptyGene_onNonEmptyGenome_end)
{
    auto genome = GenomeDescription().genes({
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
    GenomeDescriptionEditService::get().addGene(genome, 2, GeneDescription().separation(false));

    ASSERT_EQ(4, genome._genes.size());
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(i == 3 ? 0 : 1, genome._genes.at(i)._nodes.size());
    }
}

TEST_F(GenomeDescriptionEditServiceTests, addEmptyGene_withReferences)
{
    auto genome = createGenome_complexCycles();
    GenomeDescriptionEditService::get().addGene(genome, 1, GeneDescription().separation(false));

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
    auto gene = GeneDescription().separation(false).nodes({
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
    auto gene = GeneDescription().separation(false).nodes({
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
    auto gene = GeneDescription().separation(false).nodes({
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
    auto gene = GeneDescription().separation(false).nodes({
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
    auto gene = GeneDescription().separation(false).nodes({
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
    auto gene = GeneDescription().separation(false).nodes({
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
    auto gene = GeneDescription().separation(false).nodes({
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

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_emptyGenome)
{
    auto genome = GenomeDescription();
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {});
    
    EXPECT_EQ(0, subGenomes.size());
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_invalidGeneReference)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(5)),  // Invalid reference beyond genome size
            NodeDescription(),
        }),
    });
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}});

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    auto const& subGenome = subGenomes.at(0).genome;

    ASSERT_EQ(1, subGenome._genes.size());
    ASSERT_EQ(2, subGenome._genes.at(0)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, subGenome._genes.at(0)._nodes.at(0).getCellType());
    // Invalid reference should remain unchanged (the castrate method has bounds checking)
    EXPECT_EQ(5, std::get<ConstructorGenomeDescription>(subGenome._genes.at(0)._nodes.at(0)._cellTypeData)._geneIndex);
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_complexCycles)
{
    auto genome = createGenome_complexCycles();
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0, 1, 2}});

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    auto const& subGenome = subGenomes.at(0).genome;

    auto const& gene0 = subGenome._genes.at(0);
    ASSERT_EQ(3, gene0._nodes.size());
    EXPECT_EQ(3, getRefGeneIndex(gene0, 0));
    EXPECT_EQ(1, getRefGeneIndex(gene0, 1));
    EXPECT_EQ(2, getRefGeneIndex(gene0, 2));

    auto const& gene1 = subGenome._genes.at(1);
    ASSERT_EQ(4, gene1._nodes.size());
    EXPECT_EQ(3, getRefGeneIndex(gene1, 0));
    EXPECT_EQ(3, getRefGeneIndex(gene1, 1));
    EXPECT_EQ(3, getRefGeneIndex(gene1, 2));

    auto const& gene2 = subGenome._genes.at(2);
    ASSERT_EQ(5, gene2._nodes.size());
    EXPECT_EQ(3, getRefGeneIndex(gene2, 0));
    EXPECT_EQ(3, getRefGeneIndex(gene2, 1));
    EXPECT_EQ(3, getRefGeneIndex(gene2, 2));
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_subCycle)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription(),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0, 1}, {2}});

    ASSERT_EQ(2, subGenomes.size());

    {
        EXPECT_EQ(0, subGenomes.at(0).startIndex);

        auto const& subGenome = subGenomes.at(0).genome;
        ASSERT_EQ(3, subGenome._genes.size());

        auto const& gene0 = subGenome._genes.at(0);
        ASSERT_EQ(2, gene0._nodes.size());
        EXPECT_EQ(1, getRefGeneIndex(gene0, 0));

        auto const& gene1 = subGenome._genes.at(1);
        ASSERT_EQ(2, gene1._nodes.size());
        EXPECT_EQ(3, getRefGeneIndex(gene1, 0));

        auto const& gene2 = subGenome._genes.at(2);
        ASSERT_EQ(0, gene2._nodes.size());
    }
    {
        EXPECT_EQ(2, subGenomes.at(1).startIndex);

        auto const& subGenome = subGenomes.at(1).genome;
        ASSERT_EQ(3, subGenome._genes.size());

        auto const& gene0 = subGenome._genes.at(0);
        ASSERT_EQ(0, gene0._nodes.size());

        auto const& gene1 = subGenome._genes.at(1);
        ASSERT_EQ(0, gene1._nodes.size());

        auto const& gene2 = subGenome._genes.at(2);
        ASSERT_EQ(2, gene2._nodes.size());
    }
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_noCycles)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0, 1, 2}});

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    auto const& subGenome = subGenomes.at(0).genome;

    ASSERT_EQ(3, subGenome._genes.size());

    auto const& gene0 = subGenome._genes.at(0);
    ASSERT_EQ(2, gene0._nodes.size());
    EXPECT_EQ(1, getRefGeneIndex(gene0, 0));
    EXPECT_EQ(2, getRefGeneIndex(gene0, 1));

    auto const& gene1 = subGenome._genes.at(1);
    ASSERT_EQ(2, gene1._nodes.size());
    EXPECT_EQ(2, getRefGeneIndex(gene1, 0));

    auto const& gene2 = subGenome._genes.at(2);
    ASSERT_EQ(2, gene2._nodes.size());
}
