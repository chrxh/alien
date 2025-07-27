
#include <gtest/gtest.h>

#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/GenomeDescriptionInfoService.h"
#include "EngineInterface/GenomeDescription.h"

class GenomeDescriptionEditServiceTests : public ::testing::Test
{
public:
    virtual ~GenomeDescriptionEditServiceTests() = default;

protected:
    GenomeDescription createGenome_3genes_3_4_5nodes()
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

    // Helper method to create a genome with no cycles for castrate testing
    GenomeDescription createGenome_noCycles()
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

    // Helper method to create a genome with a simple cycle
    GenomeDescription createGenome_simpleCycle()
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
        });
    }

    // Helper method to create a genome with a complex cycle involving 3 genes
    GenomeDescription createGenome_complexCycle()
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
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
                NodeDescription(),
            }),
        });
    }

    // Helper method to create a genome with mixed separation flags
    GenomeDescription createGenome_mixedSeparation()
    {
        return GenomeDescription().genes({
            GeneDescription().separation(true).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription(),
            }),
            GeneDescription().separation(false).nodes({  // Referenced gene has separation=false
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
                NodeDescription(),
            }),
        });
    }

    // Helper method to create a genome where referenced gene has separation=false
    GenomeDescription createGenome_noSeparationReference()
    {
        return GenomeDescription().genes({
            GeneDescription().separation(true).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription(),
            }),
            GeneDescription().separation(false).nodes({  // No separation, should not cause castration when referenced
                NodeDescription(),
            }),
            GeneDescription().separation(true).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),  // References gene 1 which has separation=false
                NodeDescription(),
            }),
        });
    }

    // Helper method to create a genome with self-referencing constructor
    GenomeDescription createGenome_selfReference()
    {
        return GenomeDescription().genes({
            GeneDescription().separation(true).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),  // Self-reference
                NodeDescription(),
            }),
        });
    }

    // Helper method to create a gene with multiple constructors for comprehensive testing
    GenomeDescription createGenome_multipleConstructors()
    {
        return GenomeDescription().genes({
            GeneDescription().separation(true).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(2)),
                NodeDescription(),
            }),
            GeneDescription().separation(true).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),  // Creates cycle with gene 0
                NodeDescription(),
            }),
            GeneDescription().separation(false).nodes({
                NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(0)),  // Also references gene 0, but gene 2 has separation=false
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
    auto genome = createGenome_3genes_3_4_5nodes();
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
    auto genome = createGenome_3genes_3_4_5nodes();
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
    auto genome = createGenome_3genes_3_4_5nodes();
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
    auto genome = createGenome_3genes_3_4_5nodes();
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
    GenomeDescriptionEditService::get().addEmptyNode(gene, 0);

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
    GenomeDescriptionEditService::get().addEmptyNode(gene, 1);

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
    GenomeDescriptionEditService::get().addEmptyNode(gene, 2);

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

TEST_F(GenomeDescriptionEditServiceTests, castrate_noCycles)
{
    auto genome = createGenome_noCycles();
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(genome);

    ASSERT_EQ(3, genome._genes.size());
    
    // Gene 0 should still reference gene 1
    ASSERT_EQ(2, genome._genes.at(0)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(0)._nodes.at(0).getCellType());
    EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(genome._genes.at(0)._nodes.at(0)._cellTypeData)._geneIndex);
    
    // Gene 1 should still reference gene 2
    ASSERT_EQ(2, genome._genes.at(1)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(1)._nodes.at(0).getCellType());
    EXPECT_EQ(2, std::get<ConstructorGenomeDescription>(genome._genes.at(1)._nodes.at(0)._cellTypeData)._geneIndex);
}

TEST_F(GenomeDescriptionEditServiceTests, castrate_simpleCycle)
{
    auto genome = createGenome_simpleCycle();
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(genome);

    ASSERT_EQ(2, genome._genes.size());
    
    // Gene 0 should still reference gene 1 (first time visiting)
    ASSERT_EQ(2, genome._genes.at(0)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(0)._nodes.at(0).getCellType());
    EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(genome._genes.at(0)._nodes.at(0)._cellTypeData)._geneIndex);
    
    // Gene 1's constructor should be castrated (references gene 0 which was already visited and has separation=true)
    ASSERT_EQ(2, genome._genes.at(1)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(1)._nodes.at(0).getCellType());
    EXPECT_EQ(2, std::get<ConstructorGenomeDescription>(genome._genes.at(1)._nodes.at(0)._cellTypeData)._geneIndex);  // Castrated to genome.size()
}

TEST_F(GenomeDescriptionEditServiceTests, castrate_complexCycle)
{
    auto genome = createGenome_complexCycle();
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(genome);

    ASSERT_EQ(3, genome._genes.size());
    
    // Gene 0 -> Gene 1 (first visit)
    ASSERT_EQ(2, genome._genes.at(0)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(0)._nodes.at(0).getCellType());
    EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(genome._genes.at(0)._nodes.at(0)._cellTypeData)._geneIndex);
    
    // Gene 1 -> Gene 2 (first visit)
    ASSERT_EQ(2, genome._genes.at(1)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(1)._nodes.at(0).getCellType());
    EXPECT_EQ(2, std::get<ConstructorGenomeDescription>(genome._genes.at(1)._nodes.at(0)._cellTypeData)._geneIndex);
    
    // Gene 2's constructor should be castrated (references gene 0 which was already visited and has separation=true)
    ASSERT_EQ(2, genome._genes.at(2)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(2)._nodes.at(0).getCellType());
    EXPECT_EQ(3, std::get<ConstructorGenomeDescription>(genome._genes.at(2)._nodes.at(0)._cellTypeData)._geneIndex);  // Castrated to genome.size()
}

TEST_F(GenomeDescriptionEditServiceTests, castrate_mixedSeparation)
{
    auto genome = createGenome_mixedSeparation();
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(genome);

    ASSERT_EQ(2, genome._genes.size());
    
    // Gene 0 should still reference gene 1
    ASSERT_EQ(2, genome._genes.at(0)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(0)._nodes.at(0).getCellType());
    EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(genome._genes.at(0)._nodes.at(0)._cellTypeData)._geneIndex);
    
    // Gene 1's constructor SHOULD be castrated because it references gene 0 which has separation=true and was already visited
    ASSERT_EQ(2, genome._genes.at(1)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(1)._nodes.at(0).getCellType());
    EXPECT_EQ(2, std::get<ConstructorGenomeDescription>(genome._genes.at(1)._nodes.at(0)._cellTypeData)._geneIndex);  // Castrated to genome.size()
}

TEST_F(GenomeDescriptionEditServiceTests, castrate_noSeparationReference)
{
    auto genome = createGenome_noSeparationReference();
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(genome);

    ASSERT_EQ(3, genome._genes.size());
    
    // Gene 0 should still reference gene 1
    ASSERT_EQ(2, genome._genes.at(0)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(0)._nodes.at(0).getCellType());
    EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(genome._genes.at(0)._nodes.at(0)._cellTypeData)._geneIndex);
    
    // Gene 2's constructor should NOT be castrated because it references gene 1 which has separation=false
    ASSERT_EQ(2, genome._genes.at(2)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(2)._nodes.at(0).getCellType());
    EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(genome._genes.at(2)._nodes.at(0)._cellTypeData)._geneIndex);  // Not castrated
}

TEST_F(GenomeDescriptionEditServiceTests, castrate_selfReference)
{
    auto genome = createGenome_selfReference();
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(genome);

    ASSERT_EQ(1, genome._genes.size());
    
    // Self-referencing constructor should be castrated
    ASSERT_EQ(2, genome._genes.at(0)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(0)._nodes.at(0).getCellType());
    EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(genome._genes.at(0)._nodes.at(0)._cellTypeData)._geneIndex);  // Castrated to genome.size()
}

TEST_F(GenomeDescriptionEditServiceTests, castrate_emptyGenome)
{
    auto genome = GenomeDescription();
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(genome);
    
    EXPECT_EQ(0, genome._genes.size());
}

TEST_F(GenomeDescriptionEditServiceTests, castrate_invalidGeneReference)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().cellTypeData(ConstructorGenomeDescription().geneIndex(5)),  // Invalid reference beyond genome size
            NodeDescription(),
        }),
    });
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(genome);

    ASSERT_EQ(1, genome._genes.size());
    ASSERT_EQ(2, genome._genes.at(0)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(0)._nodes.at(0).getCellType());
    // Invalid reference should remain unchanged (the castrate method has bounds checking)
    EXPECT_EQ(5, std::get<ConstructorGenomeDescription>(genome._genes.at(0)._nodes.at(0)._cellTypeData)._geneIndex);
}

TEST_F(GenomeDescriptionEditServiceTests, castrate_multipleConstructors)
{
    auto genome = createGenome_multipleConstructors();
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(genome);

    ASSERT_EQ(3, genome._genes.size());
    
    // Gene 0 has two constructors: one to gene 1, one to gene 2
    ASSERT_EQ(3, genome._genes.at(0)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(0)._nodes.at(0).getCellType());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(0)._nodes.at(1).getCellType());
    EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(genome._genes.at(0)._nodes.at(0)._cellTypeData)._geneIndex);  // Still references gene 1
    EXPECT_EQ(2, std::get<ConstructorGenomeDescription>(genome._genes.at(0)._nodes.at(1)._cellTypeData)._geneIndex);  // Still references gene 2
    
    // Gene 1's constructor should be castrated (references gene 0 which was already visited and has separation=true)
    ASSERT_EQ(2, genome._genes.at(1)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(1)._nodes.at(0).getCellType());
    EXPECT_EQ(3, std::get<ConstructorGenomeDescription>(genome._genes.at(1)._nodes.at(0)._cellTypeData)._geneIndex);  // Castrated to genome.size()
    
    // Gene 2's constructor should also be castrated (references gene 0 which was already visited and has separation=true)  
    ASSERT_EQ(2, genome._genes.at(2)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, genome._genes.at(2)._nodes.at(0).getCellType());
    EXPECT_EQ(3, std::get<ConstructorGenomeDescription>(genome._genes.at(2)._nodes.at(0)._cellTypeData)._geneIndex);  // Castrated to genome.size()
}
