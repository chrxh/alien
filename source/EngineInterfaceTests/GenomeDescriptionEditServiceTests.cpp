
#include <boost/range/adaptors.hpp>

#include <gtest/gtest.h>

#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/GenomeDescriptionEditService.h>
#include <EngineInterface/GenomeDescriptionInfoService.h>

class GenomeDescriptionEditServiceTests : public ::testing::Test
{
public:
    virtual ~GenomeDescriptionEditServiceTests() = default;

protected:
    int getRefGeneIndex(GeneDescription const& gene, int nodeIndex) const
    {
        return std::get<ConstructorGenomeDescription>(gene._nodes.at(nodeIndex)._cellType)._geneIndex;
    };

    GenomeDescription createGenome_complexCycles() const
    {
        return GenomeDescription().genes({
            GeneDescription().separation(false).nodes({
                NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
                NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
            }),
            GeneDescription().separation(false).nodes({
                NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
                NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
                NodeDescription(),
            }),
            GeneDescription().separation(false).nodes({
                NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
                NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
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
            EXPECT_EQ(0, std::get<ConstructorGenomeDescription>(gene._nodes.at(0)._cellType)._geneIndex);
            EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(gene._nodes.at(1)._cellType)._geneIndex);
            EXPECT_EQ(3, std::get<ConstructorGenomeDescription>(gene._nodes.at(2)._cellType)._geneIndex);
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
        EXPECT_EQ(0, std::get<ConstructorGenomeDescription>(gene._nodes.at(0)._cellType)._geneIndex);
        EXPECT_EQ(0, std::get<ConstructorGenomeDescription>(gene._nodes.at(1)._cellType)._geneIndex);
        EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(gene._nodes.at(2)._cellType)._geneIndex);
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
        EXPECT_EQ(0, std::get<ConstructorGenomeDescription>(gene._nodes.at(0)._cellType)._geneIndex);
        EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(gene._nodes.at(1)._cellType)._geneIndex);
        EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(gene._nodes.at(2)._cellType)._geneIndex);
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
        EXPECT_EQ(0, std::get<ConstructorGenomeDescription>(gene._nodes.at(0)._cellType)._geneIndex);
        EXPECT_EQ(2, std::get<ConstructorGenomeDescription>(gene._nodes.at(1)._cellType)._geneIndex);
        EXPECT_EQ(1, std::get<ConstructorGenomeDescription>(gene._nodes.at(2)._cellType)._geneIndex);
    }
}

TEST_F(GenomeDescriptionEditServiceTests, addEmptyNode_start)
{
    auto gene = GeneDescription().separation(false).nodes({
        NodeDescription().cellType(DepotGenomeDescription()),
        NodeDescription().cellType(ConstructorGenomeDescription()),
        NodeDescription().cellType(SensorGenomeDescription()),
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
        NodeDescription().cellType(DepotGenomeDescription()),
        NodeDescription().cellType(ConstructorGenomeDescription()),
        NodeDescription().cellType(SensorGenomeDescription()),
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
        NodeDescription().cellType(DepotGenomeDescription()),
        NodeDescription().cellType(ConstructorGenomeDescription()),
        NodeDescription().cellType(SensorGenomeDescription()),
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
        NodeDescription().cellType(DepotGenomeDescription()),
        NodeDescription().cellType(ConstructorGenomeDescription()),
        NodeDescription().cellType(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().removeNode(gene, 0);

    ASSERT_EQ(2, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Constructor, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Sensor, gene._nodes.at(1).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests, removeNode_middle)
{
    auto gene = GeneDescription().separation(false).nodes({
        NodeDescription().cellType(DepotGenomeDescription()),
        NodeDescription().cellType(ConstructorGenomeDescription()),
        NodeDescription().cellType(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().removeNode(gene, 1);

    ASSERT_EQ(2, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Sensor, gene._nodes.at(1).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests, removeNode_end)
{
    auto gene = GeneDescription().separation(false).nodes({
        NodeDescription().cellType(DepotGenomeDescription()),
        NodeDescription().cellType(ConstructorGenomeDescription()),
        NodeDescription().cellType(SensorGenomeDescription()),
    });
    GenomeDescriptionEditService::get().removeNode(gene, 2);

    ASSERT_EQ(2, gene._nodes.size());
    EXPECT_EQ(CellTypeGenome_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellTypeGenome_Constructor, gene._nodes.at(1).getCellType());
}

TEST_F(GenomeDescriptionEditServiceTests, swapNodes)
{
    auto gene = GeneDescription().separation(false).nodes({
        NodeDescription().cellType(DepotGenomeDescription()),
        NodeDescription().cellType(ConstructorGenomeDescription()),
        NodeDescription().cellType(SensorGenomeDescription()),
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
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {}, false);

    EXPECT_EQ(0, subGenomes.size());
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_invalidGeneReference)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(5)),  // Invalid reference beyond genome size
            NodeDescription(),
        }),
    });
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_FALSE(subGenomes.at(0).trimmed);
    auto const& subGenome = subGenomes.at(0).genome;

    ASSERT_EQ(1, subGenome._genes.size());
    ASSERT_EQ(2, subGenome._genes.at(0)._nodes.size());
    ASSERT_EQ(CellTypeGenome_Constructor, subGenome._genes.at(0)._nodes.at(0).getCellType());
    // Invalid reference should remain unchanged (the castrate method has bounds checking)
    EXPECT_EQ(5, std::get<ConstructorGenomeDescription>(subGenome._genes.at(0)._nodes.at(0)._cellType)._geneIndex);
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_onlyBaseAndConstructor)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription()
                .cellType(ConstructorGenomeDescription())
                .neuralNetwork(NeuralNetworkGenomeDescription().weight(2, 3, 0.4f))
                .signalRestriction(SignalRestrictionGenomeDescription().mode(SignalRestrictionMode_Active).openingAngle(3.0f)),
            NodeDescription().cellType(DepotGenomeDescription()),
            NodeDescription().cellType(BaseGenomeDescription()),
            NodeDescription().cellType(SensorGenomeDescription()),
            NodeDescription().cellType(GeneratorGenomeDescription()),
            NodeDescription().cellType(AttackerGenomeDescription()),
            NodeDescription().cellType(InjectorGenomeDescription()),
            NodeDescription().cellType(MuscleGenomeDescription()),
            NodeDescription().cellType(DefenderGenomeDescription()),
            NodeDescription().cellType(ReconnectorGenomeDescription()),
            NodeDescription().cellType(DetonatorGenomeDescription()),
        }),
    });

    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}}, false);

    EXPECT_EQ(1, subGenomes.size());
    auto const& subGenome = subGenomes.at(0).genome;
    auto const& gene0 = subGenome._genes.at(0);
    for (auto const& [index, node] : gene0._nodes | boost::adaptors::indexed(0)) {
        EXPECT_EQ(index == 0 ? CellTypeGenome_Constructor : CellTypeGenome_Base, node.getCellType());
    }
    EXPECT_EQ(NeuralNetworkGenomeDescription(), gene0._nodes.front()._neuralNetwork);
    EXPECT_EQ(SignalRestrictionGenomeDescription(), gene0._nodes.front()._signalRestriction);
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_complexCycles)
{
    auto genome = createGenome_complexCycles();
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0, 1, 2}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_FALSE(subGenomes.at(0).trimmed);
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
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
            NodeDescription(),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0, 1}, {2}}, false);

    ASSERT_EQ(2, subGenomes.size());

    {
        EXPECT_EQ(0, subGenomes.at(0).startIndex);
        EXPECT_FALSE(subGenomes.at(0).trimmed);

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
        EXPECT_FALSE(subGenomes.at(1).trimmed);

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
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(2)),
            NodeDescription(),
        }),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0, 1, 2}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_FALSE(subGenomes.at(0).trimmed);
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
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0)),
        }),
    });
    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}, {1}}, false);

    ASSERT_EQ(2, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_FALSE(subGenomes.at(0).trimmed);
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

    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_FALSE(subGenomes.at(0).trimmed);
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

    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_TRUE(subGenomes.at(0).trimmed);
    auto const& subGenome = subGenomes.at(0).genome;

    // Should be trimmed by reducing concatenations
    ASSERT_EQ(1, subGenome._genes.size());
    EXPECT_EQ(5, subGenome._genes.at(0)._nodes.size());
    EXPECT_EQ(PREVIEW_MAX_CELLS / 5, subGenome._genes.at(0)._numConcatenations);

    auto resultingCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_EQ(PREVIEW_MAX_CELLS / 5 * 5, resultingCells);
}

TEST_F(GenomeDescriptionEditServiceTests, createSubGenomesForPreview_trimming_exceedsLimit_infiniteConcatenations)
{
    // Create a genome that exceeds PREVIEW_MAX_CELLS by having too many concatenations
    // Gene 0: 5 nodes, PREVIEW_MAX_CELLS / 5 + 1 concatenations = exceeds limit of PREVIEW_MAX_CELLS
    auto genome = GenomeDescription().genes({
        GeneDescription()
            .separation(false)
            .numConcatenations(std::numeric_limits<int>::max())
            .nodes({
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
            }),
    });

    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_TRUE(subGenomes.at(0).trimmed);
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

    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_TRUE(subGenomes.at(0).trimmed);
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
            genome._genes[0]._nodes.emplace_back(NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)));
        } else {
            genome._genes[0]._nodes.emplace_back(NodeDescription());
        }
    }

    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0, 1}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_TRUE(subGenomes.at(0).trimmed);
    auto const& subGenome = subGenomes.at(0).genome;

    // Should be trimmed to PREVIEW_MAX_CELLS nodes, constructor nodes should be castrated
    ASSERT_EQ(2, subGenome._genes.size());
    EXPECT_EQ(PREVIEW_MAX_CELLS, subGenome._genes.at(0)._nodes.size());
    EXPECT_EQ(1, subGenome._genes.at(0)._numConcatenations);

    // Check that constructor nodes have been castrated (gene index set beyond genome size)
    bool foundCastratedConstructor = false;
    for (auto const& node : subGenome._genes.at(0)._nodes) {
        if (node.getCellType() == CellTypeGenome_Constructor) {
            auto const& constructor = std::get<ConstructorGenomeDescription>(node._cellType);
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

    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0}, {1}}, false);

    ASSERT_EQ(2, subGenomes.size());

    // Each subgenome should be trimmed to PREVIEW_MAX_CELLS / 2 nodes
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(i, subGenomes.at(i).startIndex);
        EXPECT_TRUE(subGenomes.at(i).trimmed);
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
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
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

    auto subGenomes = GenomeDescriptionEditService::get().createSubGenomesForPreview(genome, {{0, 1}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_TRUE(subGenomes.at(0).trimmed);
    auto const& subGenome = subGenomes.at(0).genome;

    // The trimming should reduce gene 1's concatenations to fit within limit
    ASSERT_EQ(2, subGenome._genes.size());
    EXPECT_EQ(2, subGenome._genes.at(0)._nodes.size());
    EXPECT_LE(subGenome._genes.at(1)._numConcatenations, PREVIEW_MAX_CELLS / 3 + 3);  // Should be reduced

    auto resultingCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_LE(resultingCells, PREVIEW_MAX_CELLS);
}

TEST_F(GenomeDescriptionEditServiceTests, createSeedCollectionForPreview_emptySubGenomes)
{
    std::vector<SubGenomeDescription> subGenomes;

    auto result = GenomeDescriptionEditService::get().createSeedCollectionForPreview(subGenomes, std::nullopt);

    EXPECT_EQ(0, result.description._creatures.size());
    EXPECT_EQ(0, result.seedCreatureIds.size());
}

TEST_F(GenomeDescriptionEditServiceTests, createSeedCollectionForPreview_singleSubGenome_noCache)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });

    // Store original genome ID to verify it gets reassigned
    auto originalGenomeId = genome._id;

    SubGenomeDescription subGenome{genome, 0, false};
    std::vector<SubGenomeDescription> subGenomes = {subGenome};

    auto result = GenomeDescriptionEditService::get().createSeedCollectionForPreview(subGenomes, std::nullopt);

    // Should have 1 seed creature
    ASSERT_EQ(1, result.description._creatures.size());
    ASSERT_EQ(1, result.seedCreatureIds.size());

    // Check creature properties
    auto const& creature = result.description._creatures.at(0);
    EXPECT_EQ(0, creature._generation);
    EXPECT_EQ(1, result.description.getCellsForCreature(creature._id).size());
    EXPECT_EQ(result.seedCreatureIds.at(0), creature._id);

    // Verify that new IDs are assigned (not the original genome ID)
    EXPECT_NE(originalGenomeId, result.description._genomes.at(0)._id);

    // Check cell position
    auto const& cell = result.description.getCellsForCreature(creature._id).at(0);
    EXPECT_FLOAT_EQ(toFloat(PREVIEW_HEIGHT) / 2, cell._pos.x);
    EXPECT_FLOAT_EQ(toFloat(PREVIEW_HEIGHT) / 2, cell._pos.y);

    // Check genome reference
    EXPECT_EQ(creature._genomeId, result.description._genomes.at(0)._id);
}

TEST_F(GenomeDescriptionEditServiceTests, createSeedCollectionForPreview_singleSubGenome_withCache)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });

    SubGenomeDescription subGenome{genome, 0, false};

    // Create cached phenotype with seed (generation 0) and offspring (generation 1)
    Description cachedPhenotype;
    cachedPhenotype.addCreature(CreatureDescription().generation(0), {CellDescription().pos(RealVector2D{0, 0})}, genome);
    auto seedAncestorId = cachedPhenotype._creatures.at(0)._id;
    cachedPhenotype.addCreature(CreatureDescription().generation(1).ancestorId(seedAncestorId), {CellDescription().pos(RealVector2D{1, 1})}, genome);

    // Store original IDs from cache
    auto originalSeedId = cachedPhenotype._creatures.at(0)._id;
    auto originalOffspringId = cachedPhenotype._creatures.at(1)._id;
    auto originalGenomeId = cachedPhenotype._genomes.at(0)._id;

    std::vector<SubGenomeDescription> subGenomes = {subGenome};
    GenotypeToPhenotypeCache cache;
    cache.insertOrAssign(subGenome, cachedPhenotype);

    auto result = GenomeDescriptionEditService::get().createSeedCollectionForPreview(subGenomes, cache);

    // Should have both seed and offspring from cache
    ASSERT_EQ(2, result.description._creatures.size());
    ASSERT_EQ(1, result.seedCreatureIds.size());

    // Verify that IDs from cache are preserved (not reassigned)
    EXPECT_EQ(originalSeedId, result.description._creatures.at(0)._id);
    EXPECT_EQ(originalOffspringId, result.description._creatures.at(1)._id);
    EXPECT_EQ(originalGenomeId, result.description._genomes.at(0)._id);

    // Check that both generation 0 and generation 1 creatures exist in the result
    auto seedCreatureId = result.seedCreatureIds.at(0);
    bool foundGen0 = false;
    bool foundGen1 = false;
    uint64_t resultGenomeId = 0;
    for (auto const& creature : result.description._creatures) {
        if (creature._generation == 0 && creature._id == seedCreatureId) {
            foundGen0 = true;
            resultGenomeId = creature._genomeId;
        } else if (creature._generation == 1) {
            foundGen1 = true;
            // Both creatures should reference the same genome
            EXPECT_EQ(resultGenomeId, creature._genomeId);
        }
    }
    EXPECT_TRUE(foundGen0);
    EXPECT_TRUE(foundGen1);

    // Verify the genome is in the result
    ASSERT_EQ(1, result.description._genomes.size());
    EXPECT_EQ(resultGenomeId, result.description._genomes.at(0)._id);
}

TEST_F(GenomeDescriptionEditServiceTests, createSeedCollectionForPreview_singleSubGenome_withCache_offspringFirst)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });

    SubGenomeDescription subGenome{genome, 0, false};

    // Create cached phenotype with offspring (generation 1) first, then seed (generation 0)
    Description cachedPhenotype;
    cachedPhenotype._genomes.emplace_back(genome);

    auto seedCreature = CreatureDescription().generation(0).genomeId(genome._id);
    auto seedId = seedCreature._id;
    auto seedCell = CellDescription().pos(RealVector2D{0, 0}).creatureId(seedId);
    cachedPhenotype._cells.emplace_back(seedCell);

    // Add offspring first
    auto offspringCreature = CreatureDescription().generation(1).genomeId(genome._id).ancestorId(seedId);
    auto offspringId = offspringCreature._id;
    auto offspringCell = CellDescription().pos(RealVector2D{1, 1}).creatureId(offspringId);
    cachedPhenotype._creatures.emplace_back(offspringCreature);
    cachedPhenotype._cells.emplace_back(offspringCell);
    // Add seed second
    cachedPhenotype._creatures.emplace_back(seedCreature);

    // Store original IDs from cache
    auto originalGenomeId = cachedPhenotype._genomes.at(0)._id;

    std::vector<SubGenomeDescription> subGenomes = {subGenome};
    GenotypeToPhenotypeCache cache;
    cache.insertOrAssign(subGenome, cachedPhenotype);

    auto result = GenomeDescriptionEditService::get().createSeedCollectionForPreview(subGenomes, cache);

    // Should have both offspring and seed from cache
    ASSERT_EQ(2, result.description._creatures.size());
    ASSERT_EQ(1, result.seedCreatureIds.size());

    // Verify that IDs from cache are preserved (not reassigned)
    EXPECT_EQ(offspringId, result.description._creatures.at(0)._id);
    EXPECT_EQ(seedId, result.description._creatures.at(1)._id);
    EXPECT_EQ(originalGenomeId, result.description._genomes.at(0)._id);

    // Check that seed creature id points to the generation 0 creature (which is at index 1 in result)
    auto seedCreatureId = result.seedCreatureIds.at(0);
    auto const& resultOffspringCreature = result.description._creatures.at(0);
    auto const& resultSeedCreature = result.description._creatures.at(1);
    EXPECT_EQ(1, resultOffspringCreature._generation);
    EXPECT_EQ(0, resultSeedCreature._generation);
    EXPECT_EQ(seedCreatureId, resultSeedCreature._id);
}

TEST_F(GenomeDescriptionEditServiceTests, createSeedCollectionForPreview_multipleSubGenomes_noCache)
{
    auto genome1 = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
    });
    auto genome2 = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });

    // Store original genome IDs to verify they get reassigned
    auto originalGenomeId1 = genome1._id;
    auto originalGenomeId2 = genome2._id;

    SubGenomeDescription subGenome1{genome1, 0, false};
    SubGenomeDescription subGenome2{genome2, 0, false};
    std::vector<SubGenomeDescription> subGenomes = {subGenome1, subGenome2};

    auto result = GenomeDescriptionEditService::get().createSeedCollectionForPreview(subGenomes, std::nullopt);

    // Should have 2 seed creatures
    ASSERT_EQ(2, result.description._creatures.size());
    ASSERT_EQ(2, result.seedCreatureIds.size());

    // Verify that new IDs are assigned for both genomes
    EXPECT_NE(originalGenomeId1, result.description._genomes.at(0)._id);
    EXPECT_NE(originalGenomeId2, result.description._genomes.at(1)._id);

    // Check positions are different (offset by PREVIEW_HEIGHT / 2)
    // Find cells for each creature based on index in cells array
    std::vector<CellDescription> cells1, cells2;
    for (auto const& cell : result.description._cells) {
        if (cell._creatureId == result.description._creatures.at(0)._id) {
            cells1.push_back(cell);
        } else if (cell._creatureId == result.description._creatures.at(1)._id) {
            cells2.push_back(cell);
        }
    }
    ASSERT_EQ(1, cells1.size());
    ASSERT_EQ(1, cells2.size());
    auto const& cell1 = cells1.at(0);
    auto const& cell2 = cells2.at(0);

    EXPECT_FLOAT_EQ(toFloat(PREVIEW_HEIGHT) / 2, cell1._pos.x);
    EXPECT_FLOAT_EQ(toFloat(PREVIEW_HEIGHT) / 2, cell1._pos.y);
    EXPECT_FLOAT_EQ(toFloat(PREVIEW_HEIGHT), cell2._pos.x);
    EXPECT_FLOAT_EQ(toFloat(PREVIEW_HEIGHT) / 2, cell2._pos.y);

    // Check seed ids correspond to correct creatures
    EXPECT_EQ(result.seedCreatureIds.at(0), result.description._creatures.at(0)._id);
    EXPECT_EQ(result.seedCreatureIds.at(1), result.description._creatures.at(1)._id);
}

TEST_F(GenomeDescriptionEditServiceTests, createSeedCollectionForPreview_multipleSubGenomes_mixedCache)
{
    auto genome1 = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
    });
    auto genome2 = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });

    SubGenomeDescription subGenome1{genome1, 0, false};
    SubGenomeDescription subGenome2{genome2, 0, false};

    // Create cached phenotype only for first subGenome
    Description cachedPhenotype;
    cachedPhenotype.addCreature(CreatureDescription().generation(0), {CellDescription().pos(RealVector2D{0, 0})}, genome1);

    // Store original IDs from cache
    auto originalCachedCreatureId = cachedPhenotype._creatures.at(0)._id;
    auto originalCachedGenomeId = cachedPhenotype._genomes.at(0)._id;

    std::vector<SubGenomeDescription> subGenomes = {subGenome1, subGenome2};
    GenotypeToPhenotypeCache cache;
    cache.insertOrAssign(subGenome1, cachedPhenotype);

    auto result = GenomeDescriptionEditService::get().createSeedCollectionForPreview(subGenomes, cache);

    // Should have 2 creatures: 1 from cache, 1 newly created seed
    ASSERT_EQ(2, result.description._creatures.size());
    ASSERT_EQ(2, result.seedCreatureIds.size());

    // Both should be generation 0 (seeds)
    EXPECT_EQ(0, result.description._creatures.at(0)._generation);
    EXPECT_EQ(0, result.description._creatures.at(1)._generation);

    // Verify that first creature ID from cache is preserved
    EXPECT_EQ(originalCachedCreatureId, result.description._creatures.at(0)._id);

    // Verify that second creature ID is different (newly assigned)
    EXPECT_NE(originalCachedCreatureId, result.description._creatures.at(1)._id);

    // Verify that first genome ID from cache is preserved
    bool foundCachedGenome = false;
    for (auto const& genome : result.description._genomes) {
        if (genome._id == originalCachedGenomeId) {
            foundCachedGenome = true;
            break;
        }
    }
    EXPECT_TRUE(foundCachedGenome);

    // Check that there are 2 genomes in the result
    ASSERT_EQ(2, result.description._genomes.size());

    // Check that each creature references a genome in the result
    for (auto const& creature : result.description._creatures) {
        bool foundGenome = false;
        for (auto const& genome : result.description._genomes) {
            if (creature._genomeId == genome._id) {
                foundGenome = true;
                break;
            }
        }
        EXPECT_TRUE(foundGenome);
    }

    // Both seed creature ids should be in the result
    for (auto const& seedId : result.seedCreatureIds) {
        bool foundSeed = false;
        for (auto const& creature : result.description._creatures) {
            if (creature._id == seedId) {
                foundSeed = true;
                break;
            }
        }
        EXPECT_TRUE(foundSeed);
    }
}

TEST_F(GenomeDescriptionEditServiceTests, createSeedCollectionForPreview_multipleSubGenomes_allCached)
{
    auto genome1 = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
    });
    auto genome2 = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });

    SubGenomeDescription subGenome1{genome1, 0, false};
    SubGenomeDescription subGenome2{genome2, 0, false};

    // Create cached phenotypes for both subGenomes
    Description cachedPhenotype1;
    cachedPhenotype1.addCreature(CreatureDescription().generation(0), {CellDescription().pos(RealVector2D{0, 0})}, genome1);

    Description cachedPhenotype2;
    cachedPhenotype2.addCreature(CreatureDescription().generation(0), {CellDescription().pos(RealVector2D{5, 5})}, genome2);

    // Store original IDs from cache
    auto originalCreatureId1 = cachedPhenotype1._creatures.at(0)._id;
    auto originalCreatureId2 = cachedPhenotype2._creatures.at(0)._id;
    auto originalGenomeId1 = cachedPhenotype1._genomes.at(0)._id;
    auto originalGenomeId2 = cachedPhenotype2._genomes.at(0)._id;

    std::vector<SubGenomeDescription> subGenomes = {subGenome1, subGenome2};
    GenotypeToPhenotypeCache cache;
    cache.insertOrAssign(subGenome1, cachedPhenotype1);
    cache.insertOrAssign(subGenome2, cachedPhenotype2);

    auto result = GenomeDescriptionEditService::get().createSeedCollectionForPreview(subGenomes, cache);

    // Should have 2 creatures from cache
    ASSERT_EQ(2, result.description._creatures.size());
    ASSERT_EQ(2, result.seedCreatureIds.size());

    // Both should be generation 0 (seeds)
    EXPECT_EQ(0, result.description._creatures.at(0)._generation);
    EXPECT_EQ(0, result.description._creatures.at(1)._generation);

    // Verify that IDs from cache are preserved
    EXPECT_EQ(originalCreatureId1, result.description._creatures.at(0)._id);
    EXPECT_EQ(originalCreatureId2, result.description._creatures.at(1)._id);

    // Verify genome IDs are preserved
    bool foundGenome1 = false;
    bool foundGenome2 = false;
    for (auto const& genome : result.description._genomes) {
        if (genome._id == originalGenomeId1) {
            foundGenome1 = true;
        }
        if (genome._id == originalGenomeId2) {
            foundGenome2 = true;
        }
    }
    EXPECT_TRUE(foundGenome1);
    EXPECT_TRUE(foundGenome2);

    // Check that there are 2 genomes in the result
    ASSERT_EQ(2, result.description._genomes.size());
}

TEST_F(GenomeDescriptionEditServiceTests, extractPhenotypesFromPreview_emptyPreview)
{
    Description preview;
    std::vector<uint64_t> seedCreatureIds;

    auto result = GenomeDescriptionEditService::get().extractPhenotypesFromPreview(std::move(preview), seedCreatureIds);

    EXPECT_EQ(0, result.size());
}

TEST_F(GenomeDescriptionEditServiceTests, extractPhenotypesFromPreview_singleSeed_noOffspring)
{
    // Create preview with a single seed creature (generation 0)
    Description preview;
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
    });

    auto seedCreature = CreatureDescription().generation(0);
    auto seedId = seedCreature._id;
    preview.addCreature(seedCreature, {CellDescription().pos(RealVector2D{0, 0})}, genome);

    std::vector<uint64_t> seedCreatureIds = {seedId};

    auto result = GenomeDescriptionEditService::get().extractPhenotypesFromPreview(std::move(preview), seedCreatureIds);

    // Should have 1 phenotype
    ASSERT_EQ(1, result.size());

    // Should contain the seed creature
    ASSERT_EQ(1, result.at(0)._creatures.size());
    EXPECT_EQ(0, result.at(0)._creatures.at(0)._generation);
    EXPECT_EQ(seedId, result.at(0)._creatures.at(0)._id);

    // Should have the genome
    ASSERT_EQ(1, result.at(0)._genomes.size());
    EXPECT_EQ(genome._id, result.at(0)._genomes.at(0)._id);
}

TEST_F(GenomeDescriptionEditServiceTests, extractPhenotypesFromPreview_singleSeed_withOffspring)
{
    // Create preview with seed and its offspring
    Description preview;
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
    });

    auto seedCreature = CreatureDescription().generation(0);
    auto seedId = seedCreature._id;
    preview.addCreature(seedCreature, {CellDescription().pos(RealVector2D{0, 0})}, genome);

    // Add offspring (generation 1)
    preview.addCreature(CreatureDescription().generation(1).ancestorId(seedId), {CellDescription().pos(RealVector2D{1, 1})}, genome);

    std::vector<uint64_t> seedCreatureIds = {seedId};

    auto result = GenomeDescriptionEditService::get().extractPhenotypesFromPreview(std::move(preview), seedCreatureIds);

    // Should have 1 phenotype
    ASSERT_EQ(1, result.size());

    // Should contain seed + 1 offspring = 2 creatures
    ASSERT_EQ(2, result.at(0)._creatures.size());

    // Check that seed is included
    EXPECT_EQ(0, result.at(0)._creatures.at(0)._generation);
    EXPECT_EQ(seedId, result.at(0)._creatures.at(0)._id);

    // Check offspring
    EXPECT_EQ(1, result.at(0)._creatures.at(1)._generation);
    EXPECT_EQ(seedId, result.at(0)._creatures.at(1)._ancestorId);

    // Check that appropriate genome is contained in the result
    ASSERT_EQ(1, result.at(0)._genomes.size());
    EXPECT_EQ(genome._id, result.at(0)._genomes.at(0)._id);
}

TEST_F(GenomeDescriptionEditServiceTests, extractPhenotypesFromPreview_multipleSeeds_withOffspring)
{
    // Create preview with multiple seeds and their offspring
    Description preview;
    auto genome1 = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
        }),
    });
    auto genome2 = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription(),
        }),
    });

    // Seed 1
    auto seed1 = CreatureDescription().generation(0);
    auto seed1Id = seed1._id;
    preview.addCreature(seed1, {CellDescription().pos(RealVector2D{0, 0})}, genome1);

    // Offspring of seed 1
    preview.addCreature(CreatureDescription().generation(1).ancestorId(seed1Id), {CellDescription().pos(RealVector2D{1, 1})}, genome1);

    // Seed 2
    auto seed2 = CreatureDescription().generation(0);
    auto seed2Id = seed2._id;
    preview.addCreature(seed2, {CellDescription().pos(RealVector2D{10, 10})}, genome2);

    // Offspring of seed 2
    preview.addCreature(CreatureDescription().generation(1).ancestorId(seed2Id), {CellDescription().pos(RealVector2D{11, 11})}, genome2);

    std::vector<uint64_t> seedCreatureIds = {seed1Id, seed2Id};

    auto result = GenomeDescriptionEditService::get().extractPhenotypesFromPreview(std::move(preview), seedCreatureIds);

    // Should have 2 phenotypes
    ASSERT_EQ(2, result.size());

    // Phenotype 1: seed1 + 1 offspring = 2 creatures
    ASSERT_EQ(2, result.at(0)._creatures.size());
    EXPECT_EQ(0, result.at(0)._creatures.at(0)._generation);
    EXPECT_EQ(seed1Id, result.at(0)._creatures.at(0)._id);
    EXPECT_EQ(1, result.at(0)._creatures.at(1)._generation);

    // Phenotype 2: seed2 + 1 offspring = 2 creatures
    ASSERT_EQ(2, result.at(1)._creatures.size());
    EXPECT_EQ(0, result.at(1)._creatures.at(0)._generation);
    EXPECT_EQ(seed2Id, result.at(1)._creatures.at(0)._id);
    EXPECT_EQ(1, result.at(1)._creatures.at(1)._generation);

    // Check that appropriate genomes are contained in result
    ASSERT_EQ(1, result.at(0)._genomes.size());
    EXPECT_EQ(genome1._id, result.at(0)._genomes.at(0)._id);
    ASSERT_EQ(1, result.at(1)._genomes.size());
    EXPECT_EQ(genome2._id, result.at(1)._genomes.at(0)._id);
}
