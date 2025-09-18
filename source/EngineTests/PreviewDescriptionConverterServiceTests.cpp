#include <gtest/gtest.h>

#include "Base/Definitions.h"

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Description.h"
#include "EngineInterface/PreviewDescriptionConverterService.h"

#include <boost/range/combine.hpp>

#include "EngineInterface/SpaceCalculator.h"

#include "IntegrationTestFramework.h"

class PreviewDescriptionConverterServiceTests : public IntegrationTestFramework
{
public:
    PreviewDescriptionConverterServiceTests()
        : IntegrationTestFramework(std::nullopt, {100, 100})
    {}
    virtual ~PreviewDescriptionConverterServiceTests() = default;

    CellPreviewDescription const& getPreviewCell(PreviewDescription const& preview, int geneIndex, int nodeIndex)
    {
        for (auto const& cell : preview._cells) {
            if (cell._geneIndex == geneIndex && cell._nodeIndex == nodeIndex) {
                return cell;
            }
        }
        CHECK(false);
    }

    struct ConnectionCheckDescription
    {
        RealVector2D cell1;
        RealVector2D cell2;
        bool arrowToCell1 = true;
        bool arrowToCell2 = true;
    };
    void checkConnections(PreviewDescription const& preview, std::vector<ConnectionCheckDescription> const& expectedConnections)
    {
        for (auto const& expectedConnection : expectedConnections) {
            auto found = false;
            for (auto const& connection : preview._connections) {
                if (approxCompare(expectedConnection.cell1, connection._cell1) && approxCompare(expectedConnection.cell2, connection._cell2)
                        && expectedConnection.arrowToCell1 == connection._arrowToCell1 && expectedConnection.arrowToCell2 == connection._arrowToCell2) {
                    found = true;
                    break;
                }
                if (approxCompare(expectedConnection.cell2, connection._cell1) && approxCompare(expectedConnection.cell1, connection._cell2)
                    && expectedConnection.arrowToCell2 == connection._arrowToCell1 && expectedConnection.arrowToCell1 == connection._arrowToCell2) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                EXPECT_TRUE(false);
            }
        }
    }
};

TEST_F(PreviewDescriptionConverterServiceTests, convertEmptyCollection)
{
    Description input;

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(GenomeDescription(), 0, std::move(input), std::nullopt);

    EXPECT_TRUE(result.description._cells.empty());
    EXPECT_TRUE(result.description._connections.empty());
}

TEST_F(PreviewDescriptionConverterServiceTests, convertTwoCellCreature_withSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription().color(2), NodeDescription().color(3)}),
    });

    auto input = Description().creatures({
        CreatureDescription().genome(genome).cells({
            CellDescription().id(1).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(2).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(1),
        }),
    });
    input.addConnection(1, 2);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(2, result.description._cells.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto cell1 = getPreviewCell(result.description, 0, 0);
    auto cell2 = getPreviewCell(result.description, 0, 1);
    EXPECT_EQ(2, cell1._color);
    EXPECT_EQ(3, cell2._color);

    auto expectedCell1_pos = RealVector2D{0, -0.5f};
    auto expectedCell2_pos = RealVector2D{0, 0.5f};
    EXPECT_TRUE(approxCompare(expectedCell1_pos, cell1._pos));
    EXPECT_TRUE(approxCompare(expectedCell2_pos, cell2._pos));
    checkConnections(result.description, {{cell1._pos, cell2._pos}});
}

TEST_F(PreviewDescriptionConverterServiceTests, convertTwoCellCreature_withSeparation_nonRootGene)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription().color(1)}),
        GeneDescription().separation(true).nodes({NodeDescription().color(2), NodeDescription().color(3)}),
    });

    auto input = Description().creatures({
        CreatureDescription().genome(genome).cells({
            CellDescription().id(1).pos({10.0f, 10.0f}).geneIndex(1).nodeIndex(0),
            CellDescription().id(2).pos({11.0f, 10.0f}).geneIndex(1).nodeIndex(1),
        }),
    });
    input.addConnection(1, 2);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 1, std::move(input), std::nullopt);

    ASSERT_EQ(2, result.description._cells.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto cell1 = getPreviewCell(result.description, 1, 0);
    auto cell2 = getPreviewCell(result.description, 1, 1);
    EXPECT_EQ(2, cell1._color);
    EXPECT_EQ(3, cell2._color);

    auto expectedCell1_pos = RealVector2D{0, -0.5f};
    auto expectedCell2_pos = RealVector2D{0, 0.5f};
    EXPECT_TRUE(approxCompare(expectedCell1_pos, cell1._pos));
    EXPECT_TRUE(approxCompare(expectedCell2_pos, cell2._pos));
    checkConnections(result.description, {{cell1._pos, cell2._pos}});
}

TEST_F(PreviewDescriptionConverterServiceTests, convertTwoCellCreature_withoutSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({NodeDescription().color(2), NodeDescription().color(3)}),
    });
    auto input = Description().creatures({
        CreatureDescription().genome(genome).cells({
            CellDescription().id(1).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(2).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(1),
        }),
    });
    input.addConnection(1, 2);
    
    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(2, result.description._cells.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto cell1 = getPreviewCell(result.description, 0, 0);
    auto cell2 = getPreviewCell(result.description, 0, 1);
    EXPECT_EQ(2, cell1._color);
    EXPECT_EQ(3, cell2._color);

    auto expectedCell1_pos = RealVector2D{0, -0.5f};
    auto expectedCell2_pos = RealVector2D{0, 0.5f};
    EXPECT_TRUE(approxCompare(expectedCell1_pos, cell1._pos));
    EXPECT_TRUE(approxCompare(expectedCell2_pos, cell2._pos));
    checkConnections(result.description, {{cell1._pos, cell2._pos}});
}

TEST_F(PreviewDescriptionConverterServiceTests, convertTwoCellCreature_separated_withSignalRestrictions)
{
    auto constexpr BaseAngle = 10.0f;
    auto constexpr OpeningAngle = 90.0f;

    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().color(2).signalRestriction(
                SignalRestrictionGenomeDescription().active(true).baseAngle(BaseAngle).openingAngle(OpeningAngle)),
            NodeDescription().color(3),
        }),
    });

    auto input = Description().creatures({
        CreatureDescription().genome(genome).cells({
            CellDescription().id(1).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0),  // Signal restrictions are fetched from genome
            CellDescription().id(2).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(1),
        }),
    });
    input.addConnection(1, 2);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(2, result.description._cells.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto cell1 = getPreviewCell(result.description, 0, 0);
    auto cell2 = getPreviewCell(result.description, 0, 1);
    ASSERT_TRUE(cell1._signalRestriction.has_value());
    EXPECT_TRUE(approxCompare(360.0f - OpeningAngle / 2 + BaseAngle, cell1._signalRestriction->_startAngle));
    EXPECT_TRUE(approxCompare(OpeningAngle / 2 + BaseAngle, cell1._signalRestriction->_endAngle));
    checkConnections(result.description, {{cell1._pos, cell2._pos, true, false}});
}

TEST_F(PreviewDescriptionConverterServiceTests, convertThreeCellCreature)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription().color(2), NodeDescription().color(3), NodeDescription().color(4)}),
    });
    auto input = Description().creatures({
        CreatureDescription().genome(genome).cells({
            CellDescription().id(1).pos({10.0f, 9.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(2).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(1),
            CellDescription().id(3).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(2),
        }),
    });
    input.addConnection(1, 2);
    input.addConnection(2, 3);
    input.addConnection(3, 1);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(3, result.description._cells.size());
    ASSERT_EQ(3, result.description._connections.size());

    auto cell1 = getPreviewCell(result.description, 0, 0);
    auto cell2 = getPreviewCell(result.description, 0, 1);
    auto cell3 = getPreviewCell(result.description, 0, 2);
    EXPECT_EQ(2, cell1._color);
    EXPECT_EQ(3, cell2._color);
    EXPECT_EQ(4, cell3._color);

    auto oneThird = 0.333333f;
    auto expectedCell1_pos = RealVector2D{oneThird * 2, -oneThird};
    auto expectedCell2_pos = RealVector2D{-oneThird, -oneThird};
    auto expectedCell3_pos = RealVector2D{-oneThird, oneThird * 2};
    EXPECT_TRUE(approxCompare(expectedCell1_pos, cell1._pos));
    EXPECT_TRUE(approxCompare(expectedCell2_pos, cell2._pos));
    EXPECT_TRUE(approxCompare(expectedCell3_pos, cell3._pos));
    checkConnections(result.description, {{cell1._pos, cell2._pos}, {cell2._pos, cell3._pos}, {cell3._pos, cell1._pos}});
}

TEST_F(PreviewDescriptionConverterServiceTests, convertCreature_oneGene_multipleNodes_multipleConcatenations)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numConcatenations(4).nodes({NodeDescription().color(2), NodeDescription().color(3)}),
    });
    auto input = Description().creatures({
        CreatureDescription().genome(genome).cells({
            CellDescription().id(1).pos({10.0f, 4.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(2).pos({10.0f, 5.0f}).geneIndex(0).nodeIndex(1),
            CellDescription().id(3).pos({10.0f, 6.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(4).pos({10.0f, 7.0f}).geneIndex(0).nodeIndex(1),
            CellDescription().id(5).pos({10.0f, 8.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(6).pos({10.0f, 9.0f}).geneIndex(0).nodeIndex(1),
            CellDescription().id(7).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(8).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(1),
        }),
    });
    for (int i = 0; i < 7; ++i) {
        input.addConnection(i + 1, i + 2);
    }

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(8, result.description._cells.size());
    ASSERT_EQ(7, result.description._connections.size());

    std::set<RealVector2D> actualPositions;
    for (const auto& cell : result.description._cells) {
        actualPositions.insert(cell._pos);
    }
    std::set<RealVector2D> expectedPositions = {
        {3.375f, -0.125f},
        {2.375f, -0.125f},
        {1.375f, -0.125f},
        {0.375f, -0.125f},
        {-0.625f, -0.125},
        {-1.625f, -0.125},
        {-2.625f, -0.125f},
        {-2.625f, 0.875f}
    };
    ASSERT_EQ(expectedPositions.size(), actualPositions.size());
    for (auto const& [expectedPosition, actualPosition] : boost::combine(expectedPositions, actualPositions)) {
        EXPECT_TRUE(approxCompare(expectedPosition, actualPosition));
    }
}

TEST_F(PreviewDescriptionConverterServiceTests, convertCreature_oneGene_oneNode_multipleConcatenations)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numConcatenations(8).nodes({NodeDescription().color(2)}),
    });
    auto input = Description().creatures({
        CreatureDescription().genome(genome).cells({
            CellDescription().id(1).pos({10.0f, 4.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(2).pos({10.0f, 5.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(3).pos({10.0f, 6.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(4).pos({10.0f, 7.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(5).pos({10.0f, 8.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(6).pos({10.0f, 9.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(7).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(8).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(0),
        }),
    });
    for (int i = 0; i < 7; ++i) {
        input.addConnection(i + 1, i + 2);
    }

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(8, result.description._cells.size());
    ASSERT_EQ(7, result.description._connections.size());

    std::set<RealVector2D> actualPositions;
    for (const auto& cell : result.description._cells) {
        actualPositions.insert(cell._pos);
    }
    std::set<RealVector2D> expectedPositions = {
        {3.375f, -0.125f}, {2.375f, -0.125f}, {1.375f, -0.125f}, {0.375f, -0.125f}, {-0.625f, -0.125}, {-1.625f, -0.125}, {-2.625f, -0.125f}, {-2.625f, 0.875f}};
    ASSERT_EQ(expectedPositions.size(), actualPositions.size());
    for (auto const& [expectedPosition, actualPosition] : boost::combine(expectedPositions, actualPositions)) {
        EXPECT_TRUE(approxCompare(expectedPosition, actualPosition));
    }
}

TEST_F(PreviewDescriptionConverterServiceTests, convertCreature_twoGenes_oneNode_multipleConcatenations)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({NodeDescription().color(2).cellType(ConstructorGenomeDescription().geneIndex(1))}),
        GeneDescription().separation(false).numConcatenations(7).nodes({NodeDescription().color(3)}),
    });
    auto input = Description().creatures({
        CreatureDescription().genome(genome).cells({
            CellDescription().id(2).pos({10.0f, 4.0f}).geneIndex(1).nodeIndex(0),
            CellDescription().id(3).pos({10.0f, 5.0f}).geneIndex(1).nodeIndex(0),
            CellDescription().id(4).pos({10.0f, 6.0f}).geneIndex(1).nodeIndex(0),
            CellDescription().id(5).pos({10.0f, 7.0f}).geneIndex(1).nodeIndex(0),
            CellDescription().id(6).pos({10.0f, 8.0f}).geneIndex(1).nodeIndex(0),
            CellDescription().id(7).pos({10.0f, 9.0f}).geneIndex(1).nodeIndex(0),
            CellDescription().id(8).pos({10.0f, 10.0f}).geneIndex(1).nodeIndex(0),
            CellDescription().id(1).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(0).cellType(ConstructorDescription().geneIndex(1)),
        }),
    });
    input.addConnection(2, 3);
    input.addConnection(3, 4);
    input.addConnection(4, 5);
    input.addConnection(5, 6);
    input.addConnection(6, 7);
    input.addConnection(7, 8);
    input.addConnection(8, 1);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(8, result.description._cells.size());
    ASSERT_EQ(7, result.description._connections.size());

    std::set<RealVector2D> actualPositions;
    for (const auto& cell : result.description._cells) {
        actualPositions.insert(cell._pos);
    }
    std::set<RealVector2D> expectedPositions = {
        {3.375f, -0.125f}, {2.375f, -0.125f}, {1.375f, -0.125f}, {0.375f, -0.125f}, {-0.625f, -0.125}, {-1.625f, -0.125}, {-2.625f, -0.125f}, {-2.625f, 0.875f}};
    ASSERT_EQ(expectedPositions.size(), actualPositions.size());
    for (auto const& [expectedPosition, actualPosition] : boost::combine(expectedPositions, actualPositions)) {
        EXPECT_TRUE(approxCompare(expectedPosition, actualPosition));
    }
}

TEST_F(PreviewDescriptionConverterServiceTests, convertCreature_oneGenes_twoNode_multipleConcatenations)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numConcatenations(4).nodes({NodeDescription(), NodeDescription()}),
    });
    auto input = Description().creatures({
        CreatureDescription().genome(genome).cells({
            CellDescription().id(1).pos({10.0f, 4.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(2).pos({10.0f, 5.0f}).geneIndex(0).nodeIndex(1),
            CellDescription().id(3).pos({10.0f, 6.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(4).pos({10.0f, 7.0f}).geneIndex(0).nodeIndex(1),
            CellDescription().id(5).pos({10.0f, 8.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(6).pos({10.0f, 9.0f}).geneIndex(0).nodeIndex(1),
            CellDescription().id(7).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0),
            CellDescription().id(8).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(1),
        }),
    });
    input.addConnection(1, 2);
    input.addConnection(2, 3);
    input.addConnection(3, 4);
    input.addConnection(4, 5);
    input.addConnection(5, 6);
    input.addConnection(6, 7);
    input.addConnection(7, 8);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(8, result.description._cells.size());
    ASSERT_EQ(7, result.description._connections.size());

    std::set<RealVector2D> actualPositions;
    for (const auto& cell : result.description._cells) {
        actualPositions.insert(cell._pos);
    }
    std::set<RealVector2D> expectedPositions = {
        {3.375f, -0.125f}, {2.375f, -0.125f}, {1.375f, -0.125f}, {0.375f, -0.125f}, {-0.625f, -0.125}, {-1.625f, -0.125}, {-2.625f, -0.125f}, {-2.625f, 0.875f}};
    ASSERT_EQ(expectedPositions.size(), actualPositions.size());
    for (auto const& [expectedPosition, actualPosition] : boost::combine(expectedPositions, actualPositions)) {
        EXPECT_TRUE(approxCompare(expectedPosition, actualPosition));
    }
}

TEST_F(PreviewDescriptionConverterServiceTests, convertCastratedCreature_withSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1))}),
        GeneDescription().separation(true).nodes({NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0))}),
    });

    auto inputCreature1 = Description().creatures({
        CreatureDescription().genome(genome).cells({
            CellDescription().id(1).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0).cellType(ConstructorDescription().geneIndex(2)),
        }),
    });
    auto inputCreature2 = Description().creatures({
        CreatureDescription().genome(genome).cells({
            CellDescription().id(2).pos({10.0f, 10.0f}).geneIndex(1).nodeIndex(0).cellType(ConstructorDescription().geneIndex(2)),
        }),
    });
    {
        auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(inputCreature1), std::nullopt);

        ASSERT_EQ(1, result.description._cells.size());
        ASSERT_EQ(0, result.description._connections.size());

        auto cell1 = getPreviewCell(result.description, 0, 0);
        EXPECT_EQ(1, cell1._constructorGeneIndex);
    }
    {
        auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 1, std::move(inputCreature2), std::nullopt);

        ASSERT_EQ(1, result.description._cells.size());
        ASSERT_EQ(0, result.description._connections.size());

        auto cell1 = getPreviewCell(result.description, 1, 0);
        EXPECT_EQ(0, cell1._constructorGeneIndex);
    }
}

TEST_F(PreviewDescriptionConverterServiceTests, convertCastratedCreature_withoutSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1))}),
        GeneDescription().separation(false).nodes({NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0))}),
    });
    auto input = Description().creatures({
        CreatureDescription().genome(genome).cells({
            CellDescription().id(0).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(0).cellType(ConstructorDescription().geneIndex(1)),
            CellDescription().id(1).pos({10.0f, 10.0f}).geneIndex(1).nodeIndex(0).cellType(ConstructorDescription().geneIndex(2)),
        }),
    });
    input.addConnection(0, 1);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(2, result.description._cells.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto cell1 = getPreviewCell(result.description, 0, 0);
    EXPECT_FALSE(cell1._constructorGeneIndex.has_value());

    auto cell2 = getPreviewCell(result.description, 1, 0);
    EXPECT_EQ(0, cell2._constructorGeneIndex);
}
