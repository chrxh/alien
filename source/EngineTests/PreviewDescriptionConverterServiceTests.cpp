#include <gtest/gtest.h>

#include "Base/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/PreviewDescriptionConverterService.h"
#include "EngineInterface/DescriptionEditService.h"
#include "IntegrationTestFramework.h"

class PreviewDescriptionConverterServiceTests 
    : public IntegrationTestFramework
{
public:
    PreviewDescriptionConverterServiceTests()
        : IntegrationTestFramework(std::nullopt, {100, 100})
    {}
    virtual ~PreviewDescriptionConverterServiceTests() = default;

protected:
    CellDescription createCell(uint64_t id, RealVector2D pos, int color = 0, uint16_t nodeIndex = 0)
    {
        return CellDescription().id(id).pos(pos).color(color).genomeNodeIndex(nodeIndex);
    }
};

TEST_F(PreviewDescriptionConverterServiceTests, convertEmptyCollection)
{
    CollectionDescription input;
    
    auto result = PreviewDescriptionConverterService::get().convert(input);
    
    EXPECT_TRUE(result._cells.empty());
    EXPECT_TRUE(result._connections.empty());
}

TEST_F(PreviewDescriptionConverterServiceTests, convertSingleCell)
{
    CollectionDescription input;
    auto cell = createCell(1, {10.0f, 20.0f}, 3, 5);
    input.addCell(cell);
    
    auto result = PreviewDescriptionConverterService::get().convert(input);
    
    EXPECT_EQ(1, result._cells.size());
    EXPECT_EQ(0, result._connections.size());
    
    EXPECT_EQ(3, result._cells.at(0)._color);
    EXPECT_EQ(5, result._cells.at(0)._nodeIndex);
    
    EXPECT_FLOAT_EQ(0.0f, result._cells.at(0)._pos.x);
    EXPECT_FLOAT_EQ(0.0f, result._cells.at(0)._pos.y);
}

TEST_F(PreviewDescriptionConverterServiceTests, convertMultipleCells)
{
    CollectionDescription input;
    
    auto cell1 = createCell(1, {10.0f, 10.0f}, 1, 2);
    auto cell2 = createCell(2, {20.0f, 10.0f}, 2, 3);
    auto cell3 = createCell(3, {15.0f, 20.0f}, 3, 4);
    
    input.addCell(cell1);
    input.addCell(cell2);
    input.addCell(cell3);
    
    auto result = PreviewDescriptionConverterService::get().convert(input);
    
    EXPECT_EQ(3, result._cells.size());
    EXPECT_EQ(0, result._connections.size());
    
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(i + 1, result._cells.at(i)._color);
        EXPECT_EQ(i + 2, result._cells.at(i)._nodeIndex);
    }
    
    CollectionDescription previewCollection;
    for (const auto& cell : result._cells) {
        CellDescription convertedCell;
        convertedCell._pos = cell._pos;
        previewCollection.addCell(convertedCell);
    }
    
    auto center = DescriptionEditService::get().calcCenter(previewCollection);
    
    EXPECT_NEAR(0.0f, center.x, 0.001f);
    EXPECT_NEAR(0.0f, center.y, 0.001f);
}

TEST_F(PreviewDescriptionConverterServiceTests, convertCellsWithConnections)
{
    auto data = CollectionDescription().addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).color(1).genomeNodeIndex(2),
        CellDescription().id(2).pos({20.0f, 10.0f}).color(2).genomeNodeIndex(3),
    });
    data.addConnection(1, 2);
    
    auto result = PreviewDescriptionConverterService::get().convert(data);
    
    EXPECT_EQ(2, result._cells.size());
    EXPECT_EQ(1, result._connections.size());
    
    auto& connection = result._connections.at(0);
    
    EXPECT_FLOAT_EQ(-5.0f, std::min(connection._cell1.x, connection._cell2.x));
    EXPECT_FLOAT_EQ(5.0f, std::max(connection._cell1.x, connection._cell2.x));
    EXPECT_FLOAT_EQ(0.0f, connection._cell1.y);
    EXPECT_FLOAT_EQ(0.0f, connection._cell2.y);
    
    EXPECT_FALSE(connection._arrowToCell1);
    EXPECT_FALSE(connection._arrowToCell2);
}

TEST_F(PreviewDescriptionConverterServiceTests, convertCreatureCells)
{
    CollectionDescription input;
    
    CreatureDescription creature;
    creature.id(100).cells({
        CellDescription().id(1).pos({5.0f, 5.0f}).color(4).genomeNodeIndex(6),
        CellDescription().id(2).pos({15.0f, 5.0f}).color(5).genomeNodeIndex(7),
    });
    
    input._creatures.push_back(creature);
    
    auto cache = input.createCache();
    input.addConnection(1, 2, cache);
    
    auto result = PreviewDescriptionConverterService::get().convert(input);
    
    EXPECT_EQ(2, result._cells.size());
    EXPECT_EQ(1, result._connections.size());
    
    EXPECT_EQ(4, result._cells.at(0)._color);
    EXPECT_EQ(6, result._cells.at(0)._nodeIndex);
    EXPECT_EQ(5, result._cells.at(1)._color);
    EXPECT_EQ(7, result._cells.at(1)._nodeIndex);
}

TEST_F(PreviewDescriptionConverterServiceTests, convertMixedCellsAndCreatures)
{
    auto data = CollectionDescription().addCells({
        CellDescription().id(1).pos({0.0f, 0.0f}).color(1).genomeNodeIndex(1),
    });
    
    CreatureDescription creature;
    creature.id(100).cells({
        CellDescription().id(2).pos({10.0f, 0.0f}).color(2).genomeNodeIndex(2),
        CellDescription().id(3).pos({20.0f, 0.0f}).color(3).genomeNodeIndex(3),
    });
    
    data._creatures.push_back(creature);
    auto cache = data.createCache();
    data.addConnection(2, 3, cache);
    
    auto result = PreviewDescriptionConverterService::get().convert(data);
    
    EXPECT_EQ(3, result._cells.size());
    EXPECT_EQ(1, result._connections.size());
    
    RealVector2D center = {0.0f, 0.0f};
    for (const auto& cell : result._cells) {
        center.x += cell._pos.x;
        center.y += cell._pos.y;
    }
    center.x /= result._cells.size();
    center.y /= result._cells.size();
    
    EXPECT_NEAR(0.0f, center.x, 0.001f);
    EXPECT_NEAR(0.0f, center.y, 0.001f);
}

TEST_F(PreviewDescriptionConverterServiceTests, avoidDuplicateConnections)
{
    auto data = CollectionDescription().addCells({
        CellDescription().id(1).pos({0.0f, 0.0f}).color(1).genomeNodeIndex(1),
        CellDescription().id(2).pos({10.0f, 0.0f}).color(2).genomeNodeIndex(2),
        CellDescription().id(3).pos({5.0f, 10.0f}).color(3).genomeNodeIndex(3),
    });
    
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 1);
    
    auto result = PreviewDescriptionConverterService::get().convert(data);
    
    EXPECT_EQ(3, result._cells.size());
    EXPECT_EQ(3, result._connections.size());
}