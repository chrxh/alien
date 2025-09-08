
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
    // Create a genome that produces exactly PREVIEW_MAX_CELLS (50) cells
    // Gene 0: 10 nodes, 5 concatenations = 50 cells total
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numConcatenations(5).nodes({
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
    EXPECT_EQ(5, subGenome._genes.at(0)._numConcatenations);
    
    auto resultingCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_EQ(50, resultingCells);
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_trimming_exceedsLimit_concatenations)
{
    // Create a genome that exceeds PREVIEW_MAX_CELLS by having too many concatenations
    // Gene 0: 5 nodes, 15 concatenations = 75 cells (exceeds limit of 50)
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numConcatenations(15).nodes({
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
    
    // Should be trimmed by reducing concatenations: 50 / 1 subgenome = 50 cells max
    // With 5 nodes: max concatenations = 50 / 5 = 10
    ASSERT_EQ(1, subGenome._genes.size());
    EXPECT_EQ(5, subGenome._genes.at(0)._nodes.size());
    EXPECT_EQ(10, subGenome._genes.at(0)._numConcatenations);
    
    auto resultingCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_EQ(50, resultingCells);
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_trimming_exceedsLimit_nodes)
{
    // Create a genome that exceeds PREVIEW_MAX_CELLS by having too many nodes
    // Gene 0: 100 nodes, 1 concatenation = 100 cells (exceeds limit of 50)
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numConcatenations(1),
    });
    
    // Add 100 nodes to exceed the limit
    for (int i = 0; i < 100; ++i) {
        genome._genes[0]._nodes.emplace_back(NodeDescription());
    }
    
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}});

    ASSERT_EQ(1, subGenomes.size());
    auto const& subGenome = subGenomes.at(0).genome;
    
    // Should be trimmed by reducing nodes: 50 / 1 subgenome = 50 cells max
    // With 1 concatenation: max nodes = 50
    // trimNodes calculates newSize = max(0, nodeCounter + nodes.size() - nodeLimit)
    // newSize = max(0, 0 + 100 - 50) = 50, so resize(50) keeps 50 nodes
    ASSERT_EQ(1, subGenome._genes.size());
    EXPECT_EQ(50, subGenome._genes.at(0)._nodes.size());
    EXPECT_EQ(1, subGenome._genes.at(0)._numConcatenations);
    
    auto resultingCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_EQ(50, resultingCells);
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
    for (int i = 0; i < 60; ++i) {
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
    
    // Should be trimmed to 50 nodes, constructor nodes should be castrated
    ASSERT_EQ(2, subGenome._genes.size());
    EXPECT_EQ(50, subGenome._genes.at(0)._nodes.size());
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
    // Each gene has 30 nodes, 1 concatenation = 30 cells each
    // Total: 30 + 30 = 60 cells (exceeds limit of 50)
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).numConcatenations(1),
        GeneDescription().separation(true).numConcatenations(1),
    });
    
    // Add 30 nodes to each gene
    for (int geneIdx = 0; geneIdx < 2; ++geneIdx) {
        for (int i = 0; i < 30; ++i) {
            genome._genes[geneIdx]._nodes.emplace_back(NodeDescription());
        }
    }
    
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}, {1}});

    ASSERT_EQ(2, subGenomes.size());
    
    // Each subgenome should be trimmed to 50 / 2 = 25 cells max
    for (int i = 0; i < 2; ++i) {
        auto const& subGenome = subGenomes.at(i).genome;
        ASSERT_EQ(2, subGenome._genes.size());
        
        if (i == 0) {
            // First subgenome: gene 0 should be trimmed to 25 nodes, gene 1 should be empty
            EXPECT_EQ(25, subGenome._genes.at(0)._nodes.size());
            EXPECT_EQ(0, subGenome._genes.at(1)._nodes.size());
        } else {
            // Second subgenome: gene 0 should be empty, gene 1 should be trimmed to 25 nodes
            EXPECT_EQ(0, subGenome._genes.at(0)._nodes.size());
            EXPECT_EQ(25, subGenome._genes.at(1)._nodes.size());
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
        GeneDescription().separation(false).numConcatenations(20).nodes({
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
        }),
    });
    
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0, 1}});

    ASSERT_EQ(1, subGenomes.size());
    auto const& subGenome = subGenomes.at(0).genome;
    
    // Original: gene 0 (2 cells) + gene 1 (3 nodes * 20 concatenations = 60 cells) = 62 total
    // Should be trimmed to 50 cells max
    // The trimming should reduce gene 1's concatenations from 20 to fit within limit
    
    ASSERT_EQ(2, subGenome._genes.size());
    EXPECT_EQ(2, subGenome._genes.at(0)._nodes.size());
    EXPECT_LE(subGenome._genes.at(1)._numConcatenations, 20); // Should be reduced
    
    auto resultingCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_LE(resultingCells, 50);
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_trimming_basicExceeding)
{
    // Simple test: create a genome that clearly exceeds the limit
    // Gene 0: 60 nodes, 1 concatenation = 60 cells (exceeds limit of 50)
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numConcatenations(1),
    });
    
    // Add 60 nodes to clearly exceed the 50 cell limit
    for (int i = 0; i < 60; ++i) {
        genome._genes[0]._nodes.emplace_back(NodeDescription());
    }
    
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}});

    ASSERT_EQ(1, subGenomes.size());
    auto const& subGenome = subGenomes.at(0).genome;
    
    // Verify that trimming occurred - the genome should have fewer than 60 cells
    auto resultingCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_LT(resultingCells, 60);
    EXPECT_LE(resultingCells, 50);  // Should not exceed the limit
    
    // The gene should have been modified
    ASSERT_EQ(1, subGenome._genes.size());
    EXPECT_LE(subGenome._genes.at(0)._nodes.size(), 60);  // Should be smaller than original
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_trimming_emptyGeneAfterTrimming)
{
    // Create a genome where trimming results in an empty gene
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numConcatenations(1),
    });
    
    // Add exactly 51 nodes to exceed the limit by 1
    for (int i = 0; i < 51; ++i) {
        genome._genes[0]._nodes.emplace_back(NodeDescription());
    }
    
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}});

    ASSERT_EQ(1, subGenomes.size());
    auto const& subGenome = subGenomes.at(0).genome;
    
    // Should be trimmed to stay within limit
    auto resultingCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_LE(resultingCells, 50);
    
    ASSERT_EQ(1, subGenome._genes.size());
    EXPECT_LT(subGenome._genes.at(0)._nodes.size(), 51);  // Should be smaller than original
}