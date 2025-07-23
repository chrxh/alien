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
    // Helper function to create a simple cell at a given position
    CellDescription createCell(uint64_t id, RealVector2D pos, int color = 0, uint16_t nodeIndex = 0)
    {
        CellDescription cell;
        cell.id(id);
        cell.pos(pos);
        cell.color(color);
        cell.genomeNodeIndex(nodeIndex);
        return cell;
    }

    // Helper function to create a connection between two cells
    void addConnection(CellDescription& cell1, CellDescription& cell2)
    {
        ConnectionDescription connection1;
        connection1.cellId(cell2._id);
        connection1.distance(1.0f);
        connection1.angleFromPrevious(0.0f);
        cell1._connections.push_back(connection1);

        ConnectionDescription connection2;
        connection2.cellId(cell1._id);
        connection2.distance(1.0f);
        connection2.angleFromPrevious(0.0f);
        cell2._connections.push_back(connection2);
    }
};

TEST_F(PreviewDescriptionConverterServiceTests, convertEmptyCollection)
{
    CollectionDescription input;
    
    auto result = PreviewDescriptionConverterService::get().convert(input);
    
    EXPECT_TRUE(result.cells.empty());
    EXPECT_TRUE(result.connections.empty());
}

TEST_F(PreviewDescriptionConverterServiceTests, convertSingleCell)
{
    CollectionDescription input;
    auto cell = createCell(1, {10.0f, 20.0f}, 3, 5);
    input.addCell(cell);
    
    auto result = PreviewDescriptionConverterService::get().convert(input);
    
    EXPECT_EQ(1, result.cells.size());
    EXPECT_EQ(0, result.connections.size());
    
    // Check cell properties
    EXPECT_EQ(3, result.cells[0].color);
    EXPECT_EQ(5, result.cells[0].nodeIndex);
    
    // Position should be centered at {0, 0}
    EXPECT_FLOAT_EQ(0.0f, result.cells[0].pos.x);
    EXPECT_FLOAT_EQ(0.0f, result.cells[0].pos.y);
}

TEST_F(PreviewDescriptionConverterServiceTests, convertMultipleCells)
{
    CollectionDescription input;
    
    // Create cells at different positions
    auto cell1 = createCell(1, {10.0f, 10.0f}, 1, 2);
    auto cell2 = createCell(2, {20.0f, 10.0f}, 2, 3);
    auto cell3 = createCell(3, {15.0f, 20.0f}, 3, 4);
    
    input.addCell(cell1);
    input.addCell(cell2);
    input.addCell(cell3);
    
    auto result = PreviewDescriptionConverterService::get().convert(input);
    
    EXPECT_EQ(3, result.cells.size());
    EXPECT_EQ(0, result.connections.size());
    
    // Check that cells are properly converted
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(i + 1, result.cells[i].color);
        EXPECT_EQ(i + 2, result.cells[i].nodeIndex);
    }
    
    // Check centering: the center should be at {0, 0}
    // Original center was at {15, 13.33...}, so cells should be offset
    RealVector2D center = {0.0f, 0.0f};
    for (const auto& cell : result.cells) {
        center.x += cell.pos.x;
        center.y += cell.pos.y;
    }
    center.x /= result.cells.size();
    center.y /= result.cells.size();
    
    EXPECT_NEAR(0.0f, center.x, 0.001f);
    EXPECT_NEAR(0.0f, center.y, 0.001f);
}

TEST_F(PreviewDescriptionConverterServiceTests, convertCellsWithConnections)
{
    CollectionDescription input;
    
    // Create two connected cells
    auto cell1 = createCell(1, {10.0f, 10.0f}, 1, 2);
    auto cell2 = createCell(2, {20.0f, 10.0f}, 2, 3);
    
    // Add connection between cells
    addConnection(cell1, cell2);
    
    input.addCell(cell1);
    input.addCell(cell2);
    
    auto result = PreviewDescriptionConverterService::get().convert(input);
    
    EXPECT_EQ(2, result.cells.size());
    EXPECT_EQ(1, result.connections.size());
    
    // Check connection positions
    auto& connection = result.connections[0];
    
    // Connection should reference the centered positions
    // Since we have 2 cells at {10,10} and {20,10}, center is {15,10}
    // After centering, positions should be {-5,0} and {5,0}
    EXPECT_FLOAT_EQ(-5.0f, std::min(connection.cell1.x, connection.cell2.x));
    EXPECT_FLOAT_EQ(5.0f, std::max(connection.cell1.x, connection.cell2.x));
    EXPECT_FLOAT_EQ(0.0f, connection.cell1.y);
    EXPECT_FLOAT_EQ(0.0f, connection.cell2.y);
    
    // Check arrow settings (should be false as per requirements)
    EXPECT_FALSE(connection.arrowToCell1);
    EXPECT_FALSE(connection.arrowToCell2);
}

TEST_F(PreviewDescriptionConverterServiceTests, convertCreatureCells)
{
    CollectionDescription input;
    
    // Create a creature with cells
    CreatureDescription creature;
    creature.id(100);
    
    auto cell1 = createCell(1, {5.0f, 5.0f}, 4, 6);
    auto cell2 = createCell(2, {15.0f, 5.0f}, 5, 7);
    addConnection(cell1, cell2);
    
    creature.cells({cell1, cell2});
    
    input._creatures.push_back(creature);
    
    auto result = PreviewDescriptionConverterService::get().convert(input);
    
    EXPECT_EQ(2, result.cells.size());
    EXPECT_EQ(1, result.connections.size());
    
    // Check that creature cells are properly converted
    EXPECT_EQ(4, result.cells[0].color);
    EXPECT_EQ(6, result.cells[0].nodeIndex);
    EXPECT_EQ(5, result.cells[1].color);
    EXPECT_EQ(7, result.cells[1].nodeIndex);
}

TEST_F(PreviewDescriptionConverterServiceTests, convertMixedCellsAndCreatures)
{
    CollectionDescription input;
    
    // Add direct cell
    auto directCell = createCell(1, {0.0f, 0.0f}, 1, 1);
    input.addCell(directCell);
    
    // Add creature with cells
    CreatureDescription creature;
    creature.setId(100);
    
    auto creatureCell1 = createCell(2, {10.0f, 0.0f}, 2, 2);
    auto creatureCell2 = createCell(3, {20.0f, 0.0f}, 3, 3);
    addConnection(creatureCell1, creatureCell2);
    
    creature.cells({creatureCell1, creatureCell2});
    input._creatures.push_back(creature);
    
    auto result = PreviewDescriptionConverterService::get().convert(input);
    
    // Should have 3 cells total (1 direct + 2 from creature)
    EXPECT_EQ(3, result.cells.size());
    // Should have 1 connection (between the creature cells)
    EXPECT_EQ(1, result.connections.size());
    
    // Check centering: all cells should be centered around {0, 0}
    RealVector2D center = {0.0f, 0.0f};
    for (const auto& cell : result.cells) {
        center.x += cell.pos.x;
        center.y += cell.pos.y;
    }
    center.x /= result.cells.size();
    center.y /= result.cells.size();
    
    EXPECT_NEAR(0.0f, center.x, 0.001f);
    EXPECT_NEAR(0.0f, center.y, 0.001f);
}

TEST_F(PreviewDescriptionConverterServiceTests, avoidDuplicateConnections)
{
    CollectionDescription input;
    
    // Create three cells in a triangle, each connected to the others
    auto cell1 = createCell(1, {0.0f, 0.0f}, 1, 1);
    auto cell2 = createCell(2, {10.0f, 0.0f}, 2, 2);
    auto cell3 = createCell(3, {5.0f, 10.0f}, 3, 3);
    
    // Add bidirectional connections
    addConnection(cell1, cell2);
    addConnection(cell2, cell3);
    addConnection(cell3, cell1);
    
    input.addCell(cell1);
    input.addCell(cell2);
    input.addCell(cell3);
    
    auto result = PreviewDescriptionConverterService::get().convert(input);
    
    EXPECT_EQ(3, result.cells.size());
    // Should have exactly 3 connections, not 6 (no duplicates)
    EXPECT_EQ(3, result.connections.size());
}