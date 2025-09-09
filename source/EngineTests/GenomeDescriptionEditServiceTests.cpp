
#include <gtest/gtest.h>

#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/GenomeDescriptionInfoService.h"
#include "EngineInterface/GenomeDescription.h"
#include "EngineInterface/EngineConstants.h"

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
    EXPECT_EQ(2, getRefGeneIndex(gene1, 2));

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

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_separation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription(),
        }),
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
        }),
    });
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}, {1}});

    ASSERT_EQ(2, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    {
        auto const& subGenome = subGenomes.at(0).genome;

        ASSERT_EQ(2, subGenome._genes.size());

        auto const& gene0 = subGenome._genes.at(0);
        ASSERT_EQ(1, gene0._nodes.size());

        auto const& gene1 = subGenome._genes.at(1);
        ASSERT_EQ(0, gene1._nodes.size());
    }
    {
        auto const& subGenome = subGenomes.at(1).genome;

        ASSERT_EQ(2, subGenome._genes.size());

        auto const& gene0 = subGenome._genes.at(0);
        ASSERT_EQ(0, gene0._nodes.size());

        auto const& gene1 = subGenome._genes.at(1);
        ASSERT_EQ(1, gene1._nodes.size());
        EXPECT_EQ(2, getRefGeneIndex(gene1, 0));  // Castrated
    }
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_trimming_withinLimit)
{
    // Create a genome that produces exactly PREVIEW_MAX_CELLS cells
    // Gene 0: 10 nodes, PREVIEW_MAX_CELLS / 10 concatenations = PREVIEW_MAX_CELLS cells total
    auto genome = GenomeDescription().genes({
        GeneDescription()
            .separation(false)
            .numConcatenations(PREVIEW_MAX_CELLS / 10)
            .nodes({
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
            }),
    });
    
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}});

    ASSERT_EQ(1, subGenomes.size());
    auto const& subGenome = subGenomes.at(0).genome;
    
    // Should not be trimmed since it's exactly at the limit
    ASSERT_EQ(1, subGenome._genes.size());
    EXPECT_EQ(10, subGenome._genes.at(0)._nodes.size());
    EXPECT_EQ(PREVIEW_MAX_CELLS / 10, subGenome._genes.at(0)._numConcatenations);
    
    auto resultingCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_EQ(PREVIEW_MAX_CELLS, resultingCells);
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_trimming_exceedsLimit_concatenations)
{
    // Create a genome that exceeds PREVIEW_MAX_CELLS by having too many concatenations
    // Gene 0: 5 nodes, PREVIEW_MAX_CELLS / 5 + 1 concatenations = exceeds limit of PREVIEW_MAX_CELLS
    auto genome = GenomeDescription().genes({
        GeneDescription()
            .separation(false)
            .numConcatenations(PREVIEW_MAX_CELLS / 5 + 1)
            .nodes({
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
        }),
    });
    
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}});

    ASSERT_EQ(1, subGenomes.size());
    auto const& subGenome = subGenomes.at(0).genome;
    
    // Should be trimmed by reducing concatenations
    ASSERT_EQ(1, subGenome._genes.size());
    EXPECT_EQ(5, subGenome._genes.at(0)._nodes.size());
    EXPECT_EQ(PREVIEW_MAX_CELLS / 5, subGenome._genes.at(0)._numConcatenations);
    
    auto resultingCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_EQ(PREVIEW_MAX_CELLS / 5 * 5, resultingCells);
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_trimming_exceedsLimit_nodes)
{
    // Create a genome that exceeds PREVIEW_MAX_CELLS by having too many nodes
    // Gene 0: PREVIEW_MAX_CELLS + 1 nodes, 1 concatenation
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numConcatenations(1),
    });
    
    // Add 100 nodes to exceed the limit
    for (int i = 0; i < PREVIEW_MAX_CELLS + 1; ++i) {
        genome._genes[0]._nodes.emplace_back(NodeDescription());
    }
    
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}});

    ASSERT_EQ(1, subGenomes.size());
    auto const& subGenome = subGenomes.at(0).genome;
    
    // Should be trimmed by reducing nodes
    ASSERT_EQ(1, subGenome._genes.size());
    EXPECT_EQ(PREVIEW_MAX_CELLS, subGenome._genes.at(0)._nodes.size());
    EXPECT_EQ(1, subGenome._genes.at(0)._numConcatenations);
    
    auto resultingCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_EQ(PREVIEW_MAX_CELLS, resultingCells);
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_trimming_constructorCastration)
{
    // Create a genome with constructor nodes that should be castrated during trimming
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numConcatenations(1),
        GeneDescription().separation(false).numConcatenations(1).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });
    
    // Add nodes to gene 0 including constructor nodes, exceeding the limit
    for (int i = 0; i < PREVIEW_MAX_CELLS + 10; ++i) {
        if (i % 10 == 0) {
            // Add constructor nodes that reference gene 1
            genome._genes[0]._nodes.emplace_back(
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)));
        } else {
            genome._genes[0]._nodes.emplace_back(NodeDescription());
        }
    }
    
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0, 1}});

    ASSERT_EQ(1, subGenomes.size());
    auto const& subGenome = subGenomes.at(0).genome;
    
    // Should be trimmed to PREVIEW_MAX_CELLS nodes, constructor nodes should be castrated
    ASSERT_EQ(2, subGenome._genes.size());
    EXPECT_EQ(PREVIEW_MAX_CELLS, subGenome._genes.at(0)._nodes.size());
    EXPECT_EQ(1, subGenome._genes.at(0)._numConcatenations);
    
    // Check that constructor nodes have been castrated (gene index set beyond genome size)
    bool foundCastratedConstructor = false;
    for (auto const& node : subGenome._genes.at(0)._nodes) {
        if (node.getCellType() == CellTypeGenome_Constructor) {
            auto const& constructor = std::get<ConstructorGenomeDescription>(node._cellTypeData);
            if (constructor._geneIndex >= static_cast<int>(subGenome._genes.size())) {
                foundCastratedConstructor = true;
                break;
            }
        }
    }
    EXPECT_TRUE(foundCastratedConstructor);
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_trimming_multipleSubGenomes)
{
    // Create multiple subgenomes that together exceed PREVIEW_MAX_CELLS
    // Each gene has PREVIEW_MAX_CELLS / 2 + 2 nodes, 1 concatenation
    // Total: PREVIEW_MAX_CELLS + 4 exceeds limit
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).numConcatenations(1),
        GeneDescription().separation(true).numConcatenations(1),
    });
    
    // Add PREVIEW_MAX_CELLS / 2 + 2 nodes to each gene
    for (int geneIdx = 0; geneIdx < 2; ++geneIdx) {
        for (int i = 0; i < PREVIEW_MAX_CELLS / 2 + 2; ++i) {
            genome._genes[geneIdx]._nodes.emplace_back(NodeDescription());
        }
    }
    
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}, {1}});

    ASSERT_EQ(2, subGenomes.size());
    
    // Each subgenome should be trimmed to PREVIEW_MAX_CELLS / 2 nodes
    for (int i = 0; i < 2; ++i) {
        auto const& subGenome = subGenomes.at(i).genome;
        ASSERT_EQ(2, subGenome._genes.size());
        
        if (i == 0) {
            // First subgenome: gene 0 should be trimmed to 25 nodes, gene 1 should be empty
            EXPECT_EQ(PREVIEW_MAX_CELLS / 2, subGenome._genes.at(0)._nodes.size());
            EXPECT_EQ(0, subGenome._genes.at(1)._nodes.size());
        } else {
            // Second subgenome: gene 0 should be empty, gene 1 should be trimmed to 25 nodes
            EXPECT_EQ(0, subGenome._genes.at(0)._nodes.size());
            EXPECT_EQ(PREVIEW_MAX_CELLS / 2, subGenome._genes.at(1)._nodes.size());
        }
    }
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_trimming_recursiveConstructors)
{
    // Create a genome with recursive constructor references that should be trimmed
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numConcatenations(1).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
        }),
        GeneDescription()
            .separation(false)
            .numConcatenations(PREVIEW_MAX_CELLS / 3 + 3)
            .nodes({
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
            }),
    });
    
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0, 1}});

    ASSERT_EQ(1, subGenomes.size());
    auto const& subGenome = subGenomes.at(0).genome;
    
    // The trimming should reduce gene 1's concatenations to fit within limit
    ASSERT_EQ(2, subGenome._genes.size());
    EXPECT_EQ(2, subGenome._genes.at(0)._nodes.size());
    EXPECT_LE(subGenome._genes.at(1)._numConcatenations, PREVIEW_MAX_CELLS / 3 + 3);  // Should be reduced
    
    auto resultingCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_LE(resultingCells, PREVIEW_MAX_CELLS);
}
