#include <gtest/gtest.h>

#include "Base/Definitions.h"

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"

#include "EngineImpl/PreviewDescriptionConverterService.h"

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

    void checkConnections(PreviewDescription const& preview, std::vector<std::pair<RealVector2D, RealVector2D>> const& expectedConnectionPos)
    {
        for (auto const& [expectedPos1, expectedPos2] : expectedConnectionPos) {
            auto found = false;
            for (auto const& connection : preview._connections) {
                if(approxCompare(expectedPos1, connection._cell1) && approxCompare(expectedPos2, connection._cell2)
                    || approxCompare(expectedPos2, connection._cell1) && approxCompare(expectedPos1, connection._cell2)) {
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
    CollectionDescription input;

    auto result = PreviewDescriptionConverterService::get().convert(std::move(input));

    EXPECT_TRUE(result._cells.empty());
    EXPECT_TRUE(result._connections.empty());
}

TEST_F(PreviewDescriptionConverterServiceTests, convertOnlySeed)
{
    auto input = CollectionDescription().creatures({
        CreatureDescription().cells({
            CellDescription().id(0).pos({10.0f, 20.0f}).color(3).geneIndex(0).nodeIndex(0),
        }),
    });

    auto result = PreviewDescriptionConverterService::get().convert(std::move(input));

    EXPECT_EQ(0, result._cells.size()); // Seed is removed
    EXPECT_EQ(0, result._connections.size());
}

TEST_F(PreviewDescriptionConverterServiceTests, convertTwoCellCreature_separated)
{
    auto input = CollectionDescription().creatures({
        CreatureDescription().cells({
            CellDescription().id(0).pos({12.0f, 10.0f}).color(1),
        }),
        CreatureDescription().cells({
            CellDescription().id(1).pos({10.0f, 10.0f}).color(2).geneIndex(0).nodeIndex(0),
            CellDescription().id(2).pos({11.0f, 10.0f}).color(3).geneIndex(0).nodeIndex(1),
        }),
    });
    input.addConnection(1, 2);

    auto result = PreviewDescriptionConverterService::get().convert(std::move(input));

    ASSERT_EQ(2, result._cells.size());
    ASSERT_EQ(1, result._connections.size());

    auto cell1 = getPreviewCell(result, 0, 0);
    auto cell2 = getPreviewCell(result, 0, 1);
    EXPECT_EQ(2, cell1._color);
    EXPECT_EQ(3, cell2._color);

    auto expectedCell1_pos = RealVector2D{0, -0.5f};
    auto expectedCell2_pos = RealVector2D{0, 0.5f};
    EXPECT_TRUE(approxCompare(expectedCell1_pos, cell1._pos));
    EXPECT_TRUE(approxCompare(expectedCell2_pos, cell2._pos));
    checkConnections(result, {{cell1._pos, cell2._pos}});
}

TEST_F(PreviewDescriptionConverterServiceTests, convertTwoCellCreature_notSeparated)
{
    auto input = CollectionDescription().creatures({
        CreatureDescription().cells({
            CellDescription().id(0).pos({12.0f, 10.0f}),
        }),
        CreatureDescription().cells({
            CellDescription().id(1).pos({10.0f, 10.0f}).color(2).geneIndex(0).nodeIndex(0),
            CellDescription().id(2).pos({11.0f, 10.0f}).color(3).geneIndex(0).nodeIndex(1),
        }),
    });
    input.addConnection(1, 2);
    input.addConnection(2, 0);
    
    auto result = PreviewDescriptionConverterService::get().convert(std::move(input));

    ASSERT_EQ(2, result._cells.size());
    ASSERT_EQ(1, result._connections.size());

    auto cell1 = getPreviewCell(result, 0, 0);
    auto cell2 = getPreviewCell(result, 0, 1);
    EXPECT_EQ(2, cell1._color);
    EXPECT_EQ(3, cell2._color);

    auto expectedCell1_pos = RealVector2D{0, -0.5f};
    auto expectedCell2_pos = RealVector2D{0, 0.5f};
    EXPECT_TRUE(approxCompare(expectedCell1_pos, cell1._pos));
    EXPECT_TRUE(approxCompare(expectedCell2_pos, cell2._pos));
    checkConnections(result, {{cell1._pos, cell2._pos}});
}

TEST_F(PreviewDescriptionConverterServiceTests, convertThreeCellCreature)
{
    auto input = CollectionDescription().creatures({
        CreatureDescription().cells({
            CellDescription().id(0).pos({12.0f, 10.0f}).color(1),
        }),
        CreatureDescription().cells({
            CellDescription().id(1).pos({10.0f, 9.0f}).color(2).geneIndex(0).nodeIndex(0),
            CellDescription().id(2).pos({10.0f, 10.0f}).color(3).geneIndex(0).nodeIndex(1),
            CellDescription().id(3).pos({11.0f, 10.0f}).color(4).geneIndex(0).nodeIndex(2),
        }),
    });
    input.addConnection(1, 2);
    input.addConnection(2, 3);
    input.addConnection(3, 1);

    auto result = PreviewDescriptionConverterService::get().convert(std::move(input));

    ASSERT_EQ(3, result._cells.size());
    ASSERT_EQ(3, result._connections.size());

    auto cell1 = getPreviewCell(result, 0, 0);
    auto cell2 = getPreviewCell(result, 0, 1);
    auto cell3 = getPreviewCell(result, 0, 2);
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
    checkConnections(result, {{cell1._pos, cell2._pos}, {cell2._pos, cell3._pos}, {cell3._pos, cell1._pos}});
}

TEST_F(PreviewDescriptionConverterServiceTests, convertOneAndTwoCellCreature)
{
    auto input = CollectionDescription().creatures({
        CreatureDescription().cells({
            CellDescription().id(0).pos({11.0f, 10.0f}).color(1),
        }),
        CreatureDescription().cells({
            CellDescription().id(1).pos({10.0f, 10.0f}).cellTypeData(ConstructorDescription().geneIndex(1)).color(2).geneIndex(0).nodeIndex(0),
        }),
        CreatureDescription().cells({
            CellDescription().id(2).pos({9.0f, 10.0f}).color(3).geneIndex(1).nodeIndex(0),
            CellDescription().id(3).pos({9.0f, 9.0f}).color(4).geneIndex(1).nodeIndex(1),
        }),
    });
    input.addConnection(2, 3);

    auto result = PreviewDescriptionConverterService::get().convert(std::move(input));

    ASSERT_EQ(3, result._cells.size());
    ASSERT_EQ(1, result._connections.size());

    auto cell1 = getPreviewCell(result, 0, 0);
    auto cell2 = getPreviewCell(result, 1, 0);
    auto cell3 = getPreviewCell(result, 1, 1);
    EXPECT_EQ(2, cell1._color);
    EXPECT_EQ(3, cell2._color);
    EXPECT_EQ(4, cell3._color);

    EXPECT_TRUE(approxCompare(cell3._pos.x, cell1._pos.x));  // result should be rotated such that cell3 and cell1 are aligned on the x-axis
    checkConnections(result, {{cell2._pos, cell3._pos}});
}
