
#include <set>

#include <boost/range/adaptors.hpp>

#include <gtest/gtest.h>

#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/GenomeDescriptionEditService.h>
#include <EngineInterface/GenomeDescriptionInfoService.h>

class GenomeDescEditServiceTests : public ::testing::Test
{
public:
    virtual ~GenomeDescEditServiceTests() = default;

protected:
    int getRefGeneIndex(GeneDesc const& gene, int nodeIndex) const
    {
        return gene._nodes.at(nodeIndex)._constructor.value()._geneIndex;
    };

    GenomeDesc createGenome_complexCycles() const
    {
        return GenomeDesc().genes({
            GeneDesc().separation(false).nodes({
                NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0)),
                NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)),
                NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2)),
            }),
            GeneDesc().separation(false).nodes({
                NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0)),
                NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)),
                NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2)),
                NodeDesc(),
            }),
            GeneDesc().separation(false).nodes({
                NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0)),
                NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)),
                NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2)),
                NodeDesc(),
                NodeDesc(),
            }),
        });
    }
};

TEST_F(GenomeDescEditServiceTests, addEmptyGene_onEmptyGenome)
{
    auto genome = GenomeDesc();
    GenomeDescEditService::get().addGene(genome, 0, GeneDesc().separation(false));

    EXPECT_EQ(1, genome._genes.size());
}

TEST_F(GenomeDescEditServiceTests, addEmptyGene_onNonEmptyGenome_start)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
        }),
        GeneDesc().separation(false).nodes({
            NodeDesc(),
        }),
        GeneDesc().separation(false).nodes({
            NodeDesc(),
        }),
    });
    GenomeDescEditService::get().addGene(genome, 0, GeneDesc().separation(false));

    EXPECT_EQ(4, genome._genes.size());
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(i == 1 ? 0 : 1, genome._genes.at(i)._nodes.size());
    }
}

TEST_F(GenomeDescEditServiceTests, addEmptyGene_onNonEmptyGenome_end)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
        }),
        GeneDesc().separation(false).nodes({
            NodeDesc(),
        }),
        GeneDesc().separation(false).nodes({
            NodeDesc(),
        }),
    });
    GenomeDescEditService::get().addGene(genome, 2, GeneDesc().separation(false));

    ASSERT_EQ(4, genome._genes.size());
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(i == 3 ? 0 : 1, genome._genes.at(i)._nodes.size());
    }
}

TEST_F(GenomeDescEditServiceTests, addEmptyGene_withReferences)
{
    auto genome = createGenome_complexCycles();
    GenomeDescEditService::get().addGene(genome, 1, GeneDesc().separation(false));

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
            EXPECT_EQ(0, gene._nodes.at(0)._constructor.value()._geneIndex);
            EXPECT_EQ(1, gene._nodes.at(1)._constructor.value()._geneIndex);
            EXPECT_EQ(3, gene._nodes.at(2)._constructor.value()._geneIndex);
        }
    }
}

TEST_F(GenomeDescEditServiceTests, removeGene_middle)
{
    auto genome = createGenome_complexCycles();
    GenomeDescEditService::get().removeGene(genome, 1);

    ASSERT_EQ(2, genome._genes.size());
    EXPECT_EQ(3, genome._genes.at(0)._nodes.size());
    EXPECT_EQ(5, genome._genes.at(1)._nodes.size());
    for (int i = 0; i < 2; ++i) {
        auto const& gene = genome._genes.at(i);
        EXPECT_EQ(0, gene._nodes.at(0)._constructor.value()._geneIndex);
        EXPECT_EQ(0, gene._nodes.at(1)._constructor.value()._geneIndex);
        EXPECT_EQ(1, gene._nodes.at(2)._constructor.value()._geneIndex);
    }
}

TEST_F(GenomeDescEditServiceTests, removeGene_end)
{
    auto genome = createGenome_complexCycles();
    GenomeDescEditService::get().removeGene(genome, 2);

    ASSERT_EQ(2, genome._genes.size());
    EXPECT_EQ(3, genome._genes.at(0)._nodes.size());
    EXPECT_EQ(4, genome._genes.at(1)._nodes.size());
    for (int i = 0; i < 2; ++i) {
        auto const& gene = genome._genes.at(i);
        EXPECT_EQ(0, gene._nodes.at(0)._constructor.value()._geneIndex);
        EXPECT_EQ(1, gene._nodes.at(1)._constructor.value()._geneIndex);
        EXPECT_EQ(1, gene._nodes.at(2)._constructor.value()._geneIndex);
    }
}

TEST_F(GenomeDescEditServiceTests, swapGenes)
{
    auto genome = createGenome_complexCycles();
    GenomeDescEditService::get().swapGenes(genome, 1);

    ASSERT_EQ(3, genome._genes.size());
    EXPECT_EQ(3, genome._genes.at(0)._nodes.size());
    EXPECT_EQ(5, genome._genes.at(1)._nodes.size());
    EXPECT_EQ(4, genome._genes.at(2)._nodes.size());
    for (int i = 0; i < 3; ++i) {
        auto const& gene = genome._genes.at(i);
        EXPECT_EQ(0, gene._nodes.at(0)._constructor.value()._geneIndex);
        EXPECT_EQ(2, gene._nodes.at(1)._constructor.value()._geneIndex);
        EXPECT_EQ(1, gene._nodes.at(2)._constructor.value()._geneIndex);
    }
}

TEST_F(GenomeDescEditServiceTests, addEmptyNode_start)
{
    auto gene = GeneDesc().separation(false).nodes({
        NodeDesc().cellType(DepotGenomeDesc()),
        NodeDesc().constructor(ConstructorGenomeDesc()),
        NodeDesc().cellType(SensorGenomeDesc()),
    });
    GenomeDescEditService::get().addNode(gene, 0, NodeDesc());

    ASSERT_EQ(4, gene._nodes.size());
    EXPECT_EQ(CellType_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellType_Base, gene._nodes.at(1).getCellType());
    EXPECT_TRUE(gene._nodes.at(2)._constructor.has_value());
    EXPECT_EQ(CellType_Sensor, gene._nodes.at(3).getCellType());
}

TEST_F(GenomeDescEditServiceTests, addEmptyNode_middle)
{
    auto gene = GeneDesc().separation(false).nodes({
        NodeDesc().cellType(DepotGenomeDesc()),
        NodeDesc().constructor(ConstructorGenomeDesc()),
        NodeDesc().cellType(SensorGenomeDesc()),
    });
    GenomeDescEditService::get().addNode(gene, 1, NodeDesc());

    ASSERT_EQ(4, gene._nodes.size());
    EXPECT_EQ(CellType_Depot, gene._nodes.at(0).getCellType());
    EXPECT_TRUE(gene._nodes.at(1)._constructor.has_value());
    EXPECT_EQ(CellType_Base, gene._nodes.at(2).getCellType());
    EXPECT_EQ(CellType_Sensor, gene._nodes.at(3).getCellType());
}

TEST_F(GenomeDescEditServiceTests, addEmptyNode_end)
{
    auto gene = GeneDesc().separation(false).nodes({
        NodeDesc().cellType(DepotGenomeDesc()),
        NodeDesc().constructor(ConstructorGenomeDesc()),
        NodeDesc().cellType(SensorGenomeDesc()),
    });
    GenomeDescEditService::get().addNode(gene, 2, NodeDesc());

    ASSERT_EQ(4, gene._nodes.size());
    EXPECT_EQ(CellType_Depot, gene._nodes.at(0).getCellType());
    EXPECT_TRUE(gene._nodes.at(1)._constructor.has_value());
    EXPECT_EQ(CellType_Sensor, gene._nodes.at(2).getCellType());
    EXPECT_EQ(CellType_Base, gene._nodes.at(3).getCellType());
}

TEST_F(GenomeDescEditServiceTests, removeNode_start)
{
    auto gene = GeneDesc().separation(false).nodes({
        NodeDesc().cellType(DepotGenomeDesc()),
        NodeDesc().constructor(ConstructorGenomeDesc()),
        NodeDesc().cellType(SensorGenomeDesc()),
    });
    GenomeDescEditService::get().removeNode(gene, 0);

    ASSERT_EQ(2, gene._nodes.size());
    EXPECT_TRUE(gene._nodes.at(0)._constructor.has_value());
    EXPECT_EQ(CellType_Sensor, gene._nodes.at(1).getCellType());
}

TEST_F(GenomeDescEditServiceTests, removeNode_middle)
{
    auto gene = GeneDesc().separation(false).nodes({
        NodeDesc().cellType(DepotGenomeDesc()),
        NodeDesc().constructor(ConstructorGenomeDesc()),
        NodeDesc().cellType(SensorGenomeDesc()),
    });
    GenomeDescEditService::get().removeNode(gene, 1);

    ASSERT_EQ(2, gene._nodes.size());
    EXPECT_EQ(CellType_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellType_Sensor, gene._nodes.at(1).getCellType());
}

TEST_F(GenomeDescEditServiceTests, removeNode_end)
{
    auto gene = GeneDesc().separation(false).nodes({
        NodeDesc().cellType(DepotGenomeDesc()),
        NodeDesc().constructor(ConstructorGenomeDesc()),
        NodeDesc().cellType(SensorGenomeDesc()),
    });
    GenomeDescEditService::get().removeNode(gene, 2);

    ASSERT_EQ(2, gene._nodes.size());
    EXPECT_EQ(CellType_Depot, gene._nodes.at(0).getCellType());
    EXPECT_TRUE(gene._nodes.at(1)._constructor.has_value());
}

TEST_F(GenomeDescEditServiceTests, swapNodes)
{
    auto gene = GeneDesc().separation(false).nodes({
        NodeDesc().cellType(DepotGenomeDesc()),
        NodeDesc().constructor(ConstructorGenomeDesc()),
        NodeDesc().cellType(SensorGenomeDesc()),
    });
    GenomeDescEditService::get().swapNodes(gene, 1);

    ASSERT_EQ(3, gene._nodes.size());
    EXPECT_EQ(CellType_Depot, gene._nodes.at(0).getCellType());
    EXPECT_EQ(CellType_Sensor, gene._nodes.at(1).getCellType());
    EXPECT_TRUE(gene._nodes.at(2)._constructor.has_value());
}

TEST_F(GenomeDescEditServiceTests, createSubGenomesForPreview_emptyGenome)
{
    auto genome = GenomeDesc();
    auto subGenomes = GenomeDescEditService::get().createSubGenomesForPreview(genome, {}, false);

    EXPECT_EQ(0, subGenomes.size());
}

TEST_F(GenomeDescEditServiceTests, createSubGenomesForPreview_invalidGeneReference)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(5)),  // Invalid reference beyond genome size
            NodeDesc(),
        }),
    });
    auto subGenomes = GenomeDescEditService::get().createSubGenomesForPreview(genome, {{0}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_FALSE(subGenomes.at(0).trimmed);
    auto const& subGenome = subGenomes.at(0).genome;

    ASSERT_EQ(1, subGenome._genes.size());
    ASSERT_EQ(2, subGenome._genes.at(0)._nodes.size());
    ASSERT_TRUE(subGenome._genes.at(0)._nodes.at(0)._constructor.has_value());
    // Invalid reference should remain unchanged (the castrate method has bounds checking)
    EXPECT_EQ(5, subGenome._genes.at(0)._nodes.at(0)._constructor.value()._geneIndex);
}

TEST_F(GenomeDescEditServiceTests, createSubGenomesForPreview_onlyBaseAndConstructor)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc()
                .constructor(ConstructorGenomeDesc())
                .neuralNetwork(NeuralNetGenomeDesc().weight(2, 3, 0.4f)),
            NodeDesc().cellType(DepotGenomeDesc()),
            NodeDesc().cellType(BaseGenomeDesc()),
            NodeDesc().cellType(SensorGenomeDesc()),
            NodeDesc().cellType(GeneratorGenomeDesc()),
            NodeDesc().cellType(AttackerGenomeDesc()),
            NodeDesc().cellType(InjectorGenomeDesc()),
            NodeDesc().cellType(MuscleGenomeDesc()),
            NodeDesc().cellType(DefenderGenomeDesc()),
            NodeDesc().cellType(ReconnectorGenomeDesc()),
            NodeDesc().cellType(DetonatorGenomeDesc()),
        }),
    });

    auto subGenomes = GenomeDescEditService::get().createSubGenomesForPreview(genome, {{0}}, false);

    EXPECT_EQ(1, subGenomes.size());
    auto const& subGenome = subGenomes.at(0).genome;
    auto const& gene0 = subGenome._genes.at(0);
    for (auto const& [index, node] : gene0._nodes | boost::adaptors::indexed(0)) {
        EXPECT_EQ(CellType_Base, node.getCellType());
        if (index == 0) {
            EXPECT_TRUE(node._constructor.has_value());
        }
        // Cell types remain their original types in preview mode
    }
    EXPECT_EQ(NeuralNetGenomeDesc(), gene0._nodes.front()._neuralNetwork);
}

TEST_F(GenomeDescEditServiceTests, createSubGenomesForPreview_complexCycles)
{
    auto genome = createGenome_complexCycles();
    auto subGenomes = GenomeDescEditService::get().createSubGenomesForPreview(genome, {{0, 1, 2}}, false);

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

TEST_F(GenomeDescEditServiceTests, createSubGenomesForPreview_subCycle)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)),
            NodeDesc(),
        }),
        GeneDesc().separation(false).nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0)),
            NodeDesc(),
        }),
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc(),
        }),
    });
    auto subGenomes = GenomeDescEditService::get().createSubGenomesForPreview(genome, {{0, 1}, {2}}, false);

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

TEST_F(GenomeDescEditServiceTests, createSubGenomesForPreview_noCycles)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2)),
        }),
        GeneDesc().separation(false).nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2)),
            NodeDesc(),
        }),
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc(),
        }),
    });
    auto subGenomes = GenomeDescEditService::get().createSubGenomesForPreview(genome, {{0, 1, 2}}, false);

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

TEST_F(GenomeDescEditServiceTests, createSubGenomesForPreview_separation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).nodes({
            NodeDesc(),
        }),
        GeneDesc().separation(true).nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0)),
        }),
    });
    auto subGenomes = GenomeDescEditService::get().createSubGenomesForPreview(genome, {{0}, {1}}, false);

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

TEST_F(GenomeDescEditServiceTests, createSubGenomesForPreview_trimming_withinLimit)
{
    // Create a genome that produces exactly PREVIEW_MAX_CELLS cells
    // Gene 0: 10 nodes, PREVIEW_MAX_CELLS / 10 concatenations = PREVIEW_MAX_CELLS cells total
    auto genome = GenomeDesc().genes({
        GeneDesc()
            .separation(false)
            .numConcatenations(PREVIEW_MAX_CELLS / 10)
            .nodes({
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
            }),
    });

    auto subGenomes = GenomeDescEditService::get().createSubGenomesForPreview(genome, {{0}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_FALSE(subGenomes.at(0).trimmed);
    auto const& subGenome = subGenomes.at(0).genome;

    // Should not be trimmed since it's exactly at the limit
    ASSERT_EQ(1, subGenome._genes.size());
    EXPECT_EQ(10, subGenome._genes.at(0)._nodes.size());
    EXPECT_EQ(PREVIEW_MAX_CELLS / 10, subGenome._genes.at(0)._numConcatenations);

    auto resultingCells = GenomeDescInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_EQ(PREVIEW_MAX_CELLS, resultingCells);
}

TEST_F(GenomeDescEditServiceTests, createSubGenomesForPreview_trimming_exceedsLimit_concatenations)
{
    // Create a genome that exceeds PREVIEW_MAX_CELLS by having too many concatenations
    // Gene 0: 5 nodes, PREVIEW_MAX_CELLS / 5 + 1 concatenations = exceeds limit of PREVIEW_MAX_CELLS
    auto genome = GenomeDesc().genes({
        GeneDesc()
            .separation(false)
            .numConcatenations(PREVIEW_MAX_CELLS / 5 + 1)
            .nodes({
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
            }),
    });

    auto subGenomes = GenomeDescEditService::get().createSubGenomesForPreview(genome, {{0}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_TRUE(subGenomes.at(0).trimmed);
    auto const& subGenome = subGenomes.at(0).genome;

    // Should be trimmed by reducing concatenations
    ASSERT_EQ(1, subGenome._genes.size());
    EXPECT_EQ(5, subGenome._genes.at(0)._nodes.size());
    EXPECT_EQ(PREVIEW_MAX_CELLS / 5, subGenome._genes.at(0)._numConcatenations);

    auto resultingCells = GenomeDescInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_EQ(PREVIEW_MAX_CELLS / 5 * 5, resultingCells);
}

TEST_F(GenomeDescEditServiceTests, createSubGenomesForPreview_trimming_exceedsLimit_infiniteConcatenations)
{
    // Create a genome that exceeds PREVIEW_MAX_CELLS by having too many concatenations
    // Gene 0: 5 nodes, PREVIEW_MAX_CELLS / 5 + 1 concatenations = exceeds limit of PREVIEW_MAX_CELLS
    auto genome = GenomeDesc().genes({
        GeneDesc()
            .separation(false)
            .numConcatenations(std::numeric_limits<int>::max())
            .nodes({
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
            }),
    });

    auto subGenomes = GenomeDescEditService::get().createSubGenomesForPreview(genome, {{0}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_TRUE(subGenomes.at(0).trimmed);
    auto const& subGenome = subGenomes.at(0).genome;

    // Should be trimmed by reducing concatenations
    ASSERT_EQ(1, subGenome._genes.size());
    EXPECT_EQ(5, subGenome._genes.at(0)._nodes.size());
    EXPECT_EQ(PREVIEW_MAX_CELLS / 5, subGenome._genes.at(0)._numConcatenations);

    auto resultingCells = GenomeDescInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_EQ(PREVIEW_MAX_CELLS / 5 * 5, resultingCells);
}

TEST_F(GenomeDescEditServiceTests, createSubGenomesForPreview_trimming_exceedsLimit_nodes)
{
    // Create a genome that exceeds PREVIEW_MAX_CELLS by having too many nodes
    // Gene 0: PREVIEW_MAX_CELLS + 1 nodes, 1 concatenation
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).numConcatenations(1),
    });

    // Add 100 nodes to exceed the limit
    for (int i = 0; i < PREVIEW_MAX_CELLS + 1; ++i) {
        genome._genes[0]._nodes.emplace_back(NodeDesc());
    }

    auto subGenomes = GenomeDescEditService::get().createSubGenomesForPreview(genome, {{0}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_TRUE(subGenomes.at(0).trimmed);
    auto const& subGenome = subGenomes.at(0).genome;

    // Should be trimmed by reducing nodes
    ASSERT_EQ(1, subGenome._genes.size());
    EXPECT_EQ(PREVIEW_MAX_CELLS, subGenome._genes.at(0)._nodes.size());
    EXPECT_EQ(1, subGenome._genes.at(0)._numConcatenations);

    auto resultingCells = GenomeDescInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_EQ(PREVIEW_MAX_CELLS, resultingCells);
}

TEST_F(GenomeDescEditServiceTests, createSubGenomesForPreview_trimming_constructorCastration)
{
    // Create a genome with constructor nodes that should be castrated during trimming
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).numConcatenations(1),
        GeneDesc().separation(false).numConcatenations(1).nodes({
            NodeDesc(),
            NodeDesc(),
        }),
    });

    // Add nodes to gene 0 including constructor nodes, exceeding the limit
    for (int i = 0; i < PREVIEW_MAX_CELLS + 10; ++i) {
        if (i % 10 == 0) {
            // Add constructor nodes that reference gene 1
            genome._genes[0]._nodes.emplace_back(NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)));
        } else {
            genome._genes[0]._nodes.emplace_back(NodeDesc());
        }
    }

    auto subGenomes = GenomeDescEditService::get().createSubGenomesForPreview(genome, {{0, 1}}, false);

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
        if (node._constructor.has_value()) {
            auto const& constructor = node._constructor.value();
            if (constructor._geneIndex >= static_cast<int>(subGenome._genes.size())) {
                foundCastratedConstructor = true;
                break;
            }
        }
    }
    EXPECT_TRUE(foundCastratedConstructor);
}

TEST_F(GenomeDescEditServiceTests, createSubGenomesForPreview_trimming_multipleSubGenomes)
{
    // Create multiple subgenomes that together exceed PREVIEW_MAX_CELLS
    // Each gene has PREVIEW_MAX_CELLS / 2 + 2 nodes, 1 concatenation
    // Total: PREVIEW_MAX_CELLS + 4 exceeds limit
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).numConcatenations(1),
        GeneDesc().separation(true).numConcatenations(1),
    });

    // Add PREVIEW_MAX_CELLS / 2 + 2 nodes to each gene
    for (int geneIdx = 0; geneIdx < 2; ++geneIdx) {
        for (int i = 0; i < PREVIEW_MAX_CELLS / 2 + 2; ++i) {
            genome._genes[geneIdx]._nodes.emplace_back(NodeDesc());
        }
    }

    auto subGenomes = GenomeDescEditService::get().createSubGenomesForPreview(genome, {{0}, {1}}, false);

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

TEST_F(GenomeDescEditServiceTests, createSubGenomesForPreview_trimming_recursiveConstructors)
{
    // Create a genome with recursive constructor references that should be trimmed
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).numConcatenations(1).nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)),
            NodeDesc(),
        }),
        GeneDesc()
            .separation(false)
            .numConcatenations(PREVIEW_MAX_CELLS / 3 + 3)
            .nodes({
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
            }),
    });

    auto subGenomes = GenomeDescEditService::get().createSubGenomesForPreview(genome, {{0, 1}}, false);

    ASSERT_EQ(1, subGenomes.size());
    EXPECT_EQ(0, subGenomes.at(0).startIndex);
    EXPECT_TRUE(subGenomes.at(0).trimmed);
    auto const& subGenome = subGenomes.at(0).genome;

    // The trimming should reduce gene 1's concatenations to fit within limit
    ASSERT_EQ(2, subGenome._genes.size());
    EXPECT_EQ(2, subGenome._genes.at(0)._nodes.size());
    EXPECT_LE(subGenome._genes.at(1)._numConcatenations, PREVIEW_MAX_CELLS / 3 + 3);  // Should be reduced

    auto resultingCells = GenomeDescInfoService::get().getNumberOfResultingCells(subGenome);
    EXPECT_LE(resultingCells, PREVIEW_MAX_CELLS);
}

TEST_F(GenomeDescEditServiceTests, createSeedCollectionForPreview_emptySubGenomes)
{
    std::vector<SubGenomeDesc> subGenomes;

    auto result = GenomeDescEditService::get().createSeedCollectionForPreview(subGenomes, std::nullopt);

    EXPECT_EQ(0, result.description._creatures.size());
    EXPECT_EQ(0, result.seedCreatureIds.size());
}

TEST_F(GenomeDescEditServiceTests, createSeedCollectionForPreview_singleSubGenome_noCache)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc(),
        }),
    });

    // Store original genome ID to verify it gets reassigned
    auto originalGenomeId = genome._id;

    SubGenomeDesc subGenome{genome, 0, false};
    std::vector<SubGenomeDesc> subGenomes = {subGenome};

    auto result = GenomeDescEditService::get().createSeedCollectionForPreview(subGenomes, std::nullopt);

    // Should have 1 seed creature
    ASSERT_EQ(1, result.description._creatures.size());
    ASSERT_EQ(1, result.seedCreatureIds.size());

    // Check creature properties
    auto const& creature = result.description._creatures.at(0);
    EXPECT_EQ(0, creature._generation);
    EXPECT_EQ(1, result.description.getObjectsForCreature(creature._id).size());
    EXPECT_EQ(result.seedCreatureIds.at(0), creature._id);

    // Verify that new IDs are assigned (not the original genome ID)
    EXPECT_NE(originalGenomeId, result.description._genomes.at(0)._id);

    // Check cell position
    auto const& object = result.description.getObjectsForCreature(creature._id).at(0);
    EXPECT_FLOAT_EQ(toFloat(PREVIEW_HEIGHT) / 2, object._pos.x);
    EXPECT_FLOAT_EQ(toFloat(PREVIEW_HEIGHT) / 2, object._pos.y);

    // Check genome reference
    EXPECT_EQ(creature._genomeId, result.description._genomes.at(0)._id);
}

TEST_F(GenomeDescEditServiceTests, createSeedCollectionForPreview_singleSubGenome_withCache)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc(),
        }),
    });

    SubGenomeDesc subGenome{genome, 0, false};

    // Create cached phenotype with seed (generation 0) and offspring (generation 1)
    Desc cachedPhenotype;
    cachedPhenotype.addCreature({ObjectDesc().pos(RealVector2D{0, 0})}, CreatureDesc().generation(0), genome);
    auto seedAncestorId = cachedPhenotype._creatures.at(0)._id;
    cachedPhenotype.addCreature({ObjectDesc().pos(RealVector2D{1, 1})}, CreatureDesc().generation(1).ancestorId(seedAncestorId), genome);

    // Store original IDs from cache
    auto originalSeedId = cachedPhenotype._creatures.at(0)._id;
    auto originalOffspringId = cachedPhenotype._creatures.at(1)._id;
    auto originalGenomeId = cachedPhenotype._genomes.at(0)._id;

    std::vector<SubGenomeDesc> subGenomes = {subGenome};
    GenotypeToPhenotypeCache cache;
    cache.insertOrAssign(subGenome, cachedPhenotype);

    auto result = GenomeDescEditService::get().createSeedCollectionForPreview(subGenomes, cache);

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

TEST_F(GenomeDescEditServiceTests, createSeedCollectionForPreview_singleSubGenome_withCache_offspringFirst)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc(),
        }),
    });

    SubGenomeDesc subGenome{genome, 0, false};

    // Create cached phenotype with offspring (generation 1) first, then seed (generation 0)
    Desc cachedPhenotype;
    cachedPhenotype._genomes.emplace_back(genome);

    auto seedCreature = CreatureDesc().generation(0).genomeId(genome._id);
    auto seedId = seedCreature._id;
    auto seedCell = ObjectDesc().pos(RealVector2D{0, 0}).type(CellDesc().creatureId(seedId));
    cachedPhenotype._objects.emplace_back(seedCell);

    // Add offspring first
    auto offspringCreature = CreatureDesc().generation(1).genomeId(genome._id).ancestorId(seedId);
    auto offspringId = offspringCreature._id;
    auto offspringCell = ObjectDesc().pos(RealVector2D{1, 1}).type(CellDesc().creatureId(offspringId));
    cachedPhenotype._creatures.emplace_back(offspringCreature);
    cachedPhenotype._objects.emplace_back(offspringCell);
    // Add seed second
    cachedPhenotype._creatures.emplace_back(seedCreature);

    // Store original IDs from cache
    auto originalGenomeId = cachedPhenotype._genomes.at(0)._id;

    std::vector<SubGenomeDesc> subGenomes = {subGenome};
    GenotypeToPhenotypeCache cache;
    cache.insertOrAssign(subGenome, cachedPhenotype);

    auto result = GenomeDescEditService::get().createSeedCollectionForPreview(subGenomes, cache);

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

TEST_F(GenomeDescEditServiceTests, createSeedCollectionForPreview_multipleSubGenomes_noCache)
{
    auto genome1 = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
        }),
    });
    auto genome2 = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc(),
        }),
    });

    // Store original genome IDs to verify they get reassigned
    auto originalGenomeId1 = genome1._id;
    auto originalGenomeId2 = genome2._id;

    SubGenomeDesc subGenome1{genome1, 0, false};
    SubGenomeDesc subGenome2{genome2, 0, false};
    std::vector<SubGenomeDesc> subGenomes = {subGenome1, subGenome2};

    auto result = GenomeDescEditService::get().createSeedCollectionForPreview(subGenomes, std::nullopt);

    // Should have 2 seed creatures
    ASSERT_EQ(2, result.description._creatures.size());
    ASSERT_EQ(2, result.seedCreatureIds.size());

    // Verify that new IDs are assigned for both genomes
    EXPECT_NE(originalGenomeId1, result.description._genomes.at(0)._id);
    EXPECT_NE(originalGenomeId2, result.description._genomes.at(1)._id);

    // Check positions are different (offset by PREVIEW_HEIGHT / 2)
    // Find cells for each creature based on index in cells array
    std::vector<ObjectDesc> cells1, cells2;
    for (auto const& object : result.description._objects) {
        if (object.getCellRef()._creatureId == result.description._creatures.at(0)._id) {
            cells1.push_back(object);
        } else if (object.getCellRef()._creatureId == result.description._creatures.at(1)._id) {
            cells2.push_back(object);
        }
    }
    ASSERT_EQ(1, cells1.size());
    ASSERT_EQ(1, cells2.size());
    auto const& object1 = cells1.at(0);
    auto const& object2 = cells2.at(0);

    EXPECT_FLOAT_EQ(toFloat(PREVIEW_HEIGHT) / 2, object1._pos.x);
    EXPECT_FLOAT_EQ(toFloat(PREVIEW_HEIGHT) / 2, object1._pos.y);
    EXPECT_FLOAT_EQ(toFloat(PREVIEW_HEIGHT), object2._pos.x);
    EXPECT_FLOAT_EQ(toFloat(PREVIEW_HEIGHT) / 2, object2._pos.y);

    // Check seed ids correspond to correct creatures
    EXPECT_EQ(result.seedCreatureIds.at(0), result.description._creatures.at(0)._id);
    EXPECT_EQ(result.seedCreatureIds.at(1), result.description._creatures.at(1)._id);
}

TEST_F(GenomeDescEditServiceTests, createSeedCollectionForPreview_multipleSubGenomes_mixedCache)
{
    auto genome1 = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
        }),
    });
    auto genome2 = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc(),
        }),
    });

    SubGenomeDesc subGenome1{genome1, 0, false};
    SubGenomeDesc subGenome2{genome2, 0, false};

    // Create cached phenotype only for first subGenome
    Desc cachedPhenotype;
    cachedPhenotype.addCreature({ObjectDesc().pos(RealVector2D{0, 0})}, CreatureDesc().generation(0), genome1);

    // Store original IDs from cache
    auto originalCachedCreatureId = cachedPhenotype._creatures.at(0)._id;
    auto originalCachedGenomeId = cachedPhenotype._genomes.at(0)._id;

    std::vector<SubGenomeDesc> subGenomes = {subGenome1, subGenome2};
    GenotypeToPhenotypeCache cache;
    cache.insertOrAssign(subGenome1, cachedPhenotype);

    auto result = GenomeDescEditService::get().createSeedCollectionForPreview(subGenomes, cache);

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

TEST_F(GenomeDescEditServiceTests, createSeedCollectionForPreview_multipleSubGenomes_allCached)
{
    auto genome1 = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
        }),
    });
    auto genome2 = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc(),
        }),
    });

    SubGenomeDesc subGenome1{genome1, 0, false};
    SubGenomeDesc subGenome2{genome2, 0, false};

    // Create cached phenotypes for both subGenomes
    Desc cachedPhenotype1;
    cachedPhenotype1.addCreature({ObjectDesc().pos(RealVector2D{0, 0})}, CreatureDesc().generation(0), genome1);

    Desc cachedPhenotype2;
    cachedPhenotype2.addCreature({ObjectDesc().pos(RealVector2D{5, 5})}, CreatureDesc().generation(0), genome2);

    // Store original IDs from cache
    auto originalCreatureId1 = cachedPhenotype1._creatures.at(0)._id;
    auto originalCreatureId2 = cachedPhenotype2._creatures.at(0)._id;
    auto originalGenomeId1 = cachedPhenotype1._genomes.at(0)._id;
    auto originalGenomeId2 = cachedPhenotype2._genomes.at(0)._id;

    std::vector<SubGenomeDesc> subGenomes = {subGenome1, subGenome2};
    GenotypeToPhenotypeCache cache;
    cache.insertOrAssign(subGenome1, cachedPhenotype1);
    cache.insertOrAssign(subGenome2, cachedPhenotype2);

    auto result = GenomeDescEditService::get().createSeedCollectionForPreview(subGenomes, cache);

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

TEST_F(GenomeDescEditServiceTests, extractPhenotypesFromPreview_emptyPreview)
{
    Desc preview;
    std::vector<uint64_t> seedCreatureIds;

    auto result = GenomeDescEditService::get().extractPhenotypesFromPreview(std::move(preview), seedCreatureIds);

    EXPECT_EQ(0, result.size());
}

TEST_F(GenomeDescEditServiceTests, extractPhenotypesFromPreview_singleSeed_noOffspring)
{
    // Create preview with a single seed creature (generation 0)
    Desc preview;
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
        }),
    });

    auto seedCreature = CreatureDesc().generation(0);
    auto seedId = seedCreature._id;
    preview.addCreature({ObjectDesc().pos(RealVector2D{0, 0})}, seedCreature, genome);

    std::vector<uint64_t> seedCreatureIds = {seedId};

    auto result = GenomeDescEditService::get().extractPhenotypesFromPreview(std::move(preview), seedCreatureIds);

    // Should have 1 phenotype
    ASSERT_EQ(1, result.size());

    // Should contain the seed creature
    ASSERT_EQ(1, result.at(0)._creatures.size());
    EXPECT_EQ(0, result.at(0)._creatures.at(0)._generation);
    EXPECT_EQ(seedId, result.at(0)._creatures.at(0)._id);

    // Should have the genome
    ASSERT_EQ(1, result.at(0)._genomes.size());
    EXPECT_EQ(genome._id, result.at(0)._genomes.at(0)._id);

    // Should have the cell associated with the seed creature
    ASSERT_EQ(1, result.at(0)._objects.size());
    EXPECT_EQ(seedId, result.at(0)._objects.at(0).getCellRef()._creatureId);
}

TEST_F(GenomeDescEditServiceTests, extractPhenotypesFromPreview_singleSeed_withOffspring)
{
    // Create preview with seed and its offspring
    Desc preview;
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
        }),
    });

    auto seedCreature = CreatureDesc().generation(0);
    auto seedId = seedCreature._id;
    preview.addCreature({ObjectDesc().pos(RealVector2D{0, 0})}, seedCreature, genome);

    // Add offspring (generation 1)
    auto offspringCreature = CreatureDesc().generation(1).ancestorId(seedId);
    auto offspringId = offspringCreature._id;
    preview.addCreature({ObjectDesc().pos(RealVector2D{1, 1})}, offspringCreature, genome);

    std::vector<uint64_t> seedCreatureIds = {seedId};

    auto result = GenomeDescEditService::get().extractPhenotypesFromPreview(std::move(preview), seedCreatureIds);

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

    // Should have 2 cells (1 from seed, 1 from offspring)
    ASSERT_EQ(2, result.at(0)._objects.size());

    // Verify cells are associated with correct creatures
    std::set<uint64_t> cellCreatureIds;
    for (auto const& object : result.at(0)._objects) {
        cellCreatureIds.insert(object.getCellRef()._creatureId);
    }
    EXPECT_TRUE(cellCreatureIds.contains(seedId));
    EXPECT_TRUE(cellCreatureIds.contains(offspringId));
}

TEST_F(GenomeDescEditServiceTests, extractPhenotypesFromPreview_multipleSeeds_withOffspring)
{
    // Create preview with multiple seeds and their offspring
    Desc preview;
    auto genome1 = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
        }),
    });
    auto genome2 = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc(),
        }),
    });

    // Seed 1
    auto seed1 = CreatureDesc().generation(0);
    auto seed1Id = seed1._id;
    preview.addCreature({ObjectDesc().pos(RealVector2D{0, 0})}, seed1, genome1);

    // Offspring of seed 1
    auto offspring1 = CreatureDesc().generation(1).ancestorId(seed1Id);
    auto offspring1Id = offspring1._id;
    preview.addCreature({ObjectDesc().pos(RealVector2D{1, 1})}, offspring1, genome1);

    // Seed 2
    auto seed2 = CreatureDesc().generation(0);
    auto seed2Id = seed2._id;
    preview.addCreature({ObjectDesc().pos(RealVector2D{10, 10})}, seed2, genome2);

    // Offspring of seed 2
    auto offspring2 = CreatureDesc().generation(1).ancestorId(seed2Id);
    auto offspring2Id = offspring2._id;
    preview.addCreature({ObjectDesc().pos(RealVector2D{11, 11})}, offspring2, genome2);

    std::vector<uint64_t> seedCreatureIds = {seed1Id, seed2Id};

    auto result = GenomeDescEditService::get().extractPhenotypesFromPreview(std::move(preview), seedCreatureIds);

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

    // Phenotype 1: should have 2 cells (1 from seed1, 1 from offspring1)
    ASSERT_EQ(2, result.at(0)._objects.size());
    std::set<uint64_t> phenotype1CellCreatureIds;
    for (auto const& object : result.at(0)._objects) {
        phenotype1CellCreatureIds.insert(object.getCellRef()._creatureId);
    }
    EXPECT_TRUE(phenotype1CellCreatureIds.contains(seed1Id));
    EXPECT_TRUE(phenotype1CellCreatureIds.contains(offspring1Id));

    // Phenotype 2: should have 2 cells (1 from seed2, 1 from offspring2)
    ASSERT_EQ(2, result.at(1)._objects.size());
    std::set<uint64_t> phenotype2CellCreatureIds;
    for (auto const& object : result.at(1)._objects) {
        phenotype2CellCreatureIds.insert(object.getCellRef()._creatureId);
    }
    EXPECT_TRUE(phenotype2CellCreatureIds.contains(seed2Id));
    EXPECT_TRUE(phenotype2CellCreatureIds.contains(offspring2Id));
}
