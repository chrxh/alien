/*
 * Tests for DescriptionEditService::flattenTopology function
 * 
 * The flattenTopology function corrects cell positions in a wrapped/toroidal world
 * to ensure that connected cells are positioned correctly relative to each other.
 * When cells are connected but positioned on opposite sides of the world boundaries,
 * the function applies topology corrections to move them to the same "side".
 *
 * Test coverage includes:
 * - Basic scenarios: empty, single cell, connected pairs
 * - World boundary wrapping corrections (horizontal, vertical, both)
 * - Complex topologies: chains, rings, disconnected components
 * - Edge cases: unconnected cells, mixed scenarios
 * - Integration with creatures: cells within CreatureDescription objects
 */

#include <gtest/gtest.h>
#include <map>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Description.h"
#include "EngineInterface/SpaceCalculator.h"
#include "Base/Vector2D.h"

class DescriptionEditServiceTests : public ::testing::Test
{
public:
    virtual ~DescriptionEditServiceTests() = default;

protected:
    // Helper to create a simple connected pair of cells
    Description createConnectedPair(RealVector2D pos1, RealVector2D pos2, uint64_t id1 = 1, uint64_t id2 = 2) const
    {
        auto cell1 = CellDescription().id(id1).pos(pos1).connections({ConnectionDescription().cellId(id2)});
        auto cell2 = CellDescription().id(id2).pos(pos2).connections({ConnectionDescription().cellId(id1)});
        
        return Description().cells({cell1, cell2});
    }

    // Helper to create a chain of connected cells
    Description createChain(std::vector<RealVector2D> positions) const
    {
        std::vector<CellDescription> cells;
        
        for (size_t i = 0; i < positions.size(); ++i) {
            uint64_t id = i + 1;
            std::vector<ConnectionDescription> connections;
            
            if (i > 0) {
                connections.push_back(ConnectionDescription().cellId(id - 1));
            }
            if (i < positions.size() - 1) {
                connections.push_back(ConnectionDescription().cellId(id + 1));
            }
            
            cells.push_back(CellDescription().id(id).pos(positions[i]).connections(connections));
        }
        
        return Description().cells(cells);
    }

    // Helper to create multiple disconnected components
    Description createDisconnectedComponents(
        std::vector<RealVector2D> component1,
        std::vector<RealVector2D> component2) const
    {
        auto desc1 = createChain(component1);
        auto desc2 = createChain(component2);
        
        // Adjust IDs in second component to avoid conflicts
        for (auto& cell : desc2._cells) {
            cell._id += component1.size();
            for (auto& conn : cell._connections) {
                conn._cellId += component1.size();
            }
        }
        
        Description result = desc1;
        result._cells.insert(result._cells.end(), desc2._cells.begin(), desc2._cells.end());
        return result;
    }
};

TEST_F(DescriptionEditServiceTests, flattenTopology_emptyCells)
{
    Description data;
    IntVector2D worldSize{100, 100};
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    EXPECT_TRUE(data._cells.empty());
}

TEST_F(DescriptionEditServiceTests, flattenTopology_singleCell)
{
    auto originalPos = RealVector2D{50.0f, 60.0f};
    Description data = Description().cells({
        CellDescription().id(1).pos(originalPos)
    });
    IntVector2D worldSize{100, 100};
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    ASSERT_EQ(1, data._cells.size());
    EXPECT_EQ(originalPos.x, data._cells[0]._pos.x);
    EXPECT_EQ(originalPos.y, data._cells[0]._pos.y);
}

TEST_F(DescriptionEditServiceTests, flattenTopology_connectedPair_noCorrection)
{
    // Two cells close together, no wrapping needed
    Description data = createConnectedPair({30.0f, 40.0f}, {32.0f, 42.0f});
    IntVector2D worldSize{100, 100};
    
    auto originalPositions = std::vector<RealVector2D>{data._cells[0]._pos, data._cells[1]._pos};
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    ASSERT_EQ(2, data._cells.size());
    // Positions should remain unchanged since no correction is needed
    EXPECT_FLOAT_EQ(originalPositions[0].x, data._cells[0]._pos.x);
    EXPECT_FLOAT_EQ(originalPositions[0].y, data._cells[0]._pos.y);
    EXPECT_FLOAT_EQ(originalPositions[1].x, data._cells[1]._pos.x);
    EXPECT_FLOAT_EQ(originalPositions[1].y, data._cells[1]._pos.y);
}

TEST_F(DescriptionEditServiceTests, flattenTopology_connectedPair_horizontalWrapping)
{
    // Two cells on opposite sides of the world horizontally
    Description data = createConnectedPair({5.0f, 50.0f}, {95.0f, 50.0f});
    IntVector2D worldSize{100, 100};
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    ASSERT_EQ(2, data._cells.size());
    // One cell should be corrected to be on the same "side" as the other
    // The exact correction depends on which cell is processed first, but they should be closer
    auto pos1 = data._cells[0]._pos;
    auto pos2 = data._cells[1]._pos;
    
    // After flattening, the cells should either be both on left side or both on right side
    bool bothOnLeft = (pos1.x < 50.0f && pos2.x < 50.0f);
    bool bothOnRight = (pos1.x > 50.0f && pos2.x > 50.0f);
    EXPECT_TRUE(bothOnLeft || bothOnRight);
}

TEST_F(DescriptionEditServiceTests, flattenTopology_connectedPair_verticalWrapping)
{
    // Two cells on opposite sides of the world vertically
    Description data = createConnectedPair({50.0f, 5.0f}, {50.0f, 95.0f});
    IntVector2D worldSize{100, 100};
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    ASSERT_EQ(2, data._cells.size());
    // After flattening, cells should be on the same "side"
    auto pos1 = data._cells[0]._pos;
    auto pos2 = data._cells[1]._pos;
    
    bool bothOnTop = (pos1.y < 50.0f && pos2.y < 50.0f);
    bool bothOnBottom = (pos1.y > 50.0f && pos2.y > 50.0f);
    EXPECT_TRUE(bothOnTop || bothOnBottom);
}

TEST_F(DescriptionEditServiceTests, flattenTopology_chain_wrapping)
{
    // Chain of cells where some wrap around the world boundary
    Description data = createChain({
        {95.0f, 50.0f},  // Near right edge
        {5.0f, 50.0f},   // Near left edge (should be corrected)
        {15.0f, 50.0f}   // Further from left edge (should be corrected)
    });
    IntVector2D worldSize{100, 100};
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    ASSERT_EQ(3, data._cells.size());
    
    // After flattening, all cells should be on the same logical side
    // They should form a reasonable chain without large gaps
    auto pos1 = data._cells[0]._pos;
    auto pos2 = data._cells[1]._pos; 
    auto pos3 = data._cells[2]._pos;
    
    // Check that positions form a logical sequence (allowing for corrections)
    float dist12 = std::abs(pos2.x - pos1.x);
    float dist23 = std::abs(pos3.x - pos2.x);
    
    // Distances should be reasonable (much less than world size)
    EXPECT_LT(dist12, 50.0f);
    EXPECT_LT(dist23, 50.0f);
}

TEST_F(DescriptionEditServiceTests, flattenTopology_disconnectedComponents)
{
    // Two separate components that shouldn't affect each other
    Description data = createDisconnectedComponents(
        {{10.0f, 10.0f}, {12.0f, 10.0f}},  // Component 1: two cells close together
        {{90.0f, 90.0f}, {92.0f, 90.0f}}   // Component 2: two cells close together
    );
    IntVector2D worldSize{100, 100};
    
    auto originalPositions = std::vector<RealVector2D>();
    for (const auto& cell : data._cells) {
        originalPositions.push_back(cell._pos);
    }
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    ASSERT_EQ(4, data._cells.size());
    
    // Since both components are internally consistent, positions should be unchanged
    for (size_t i = 0; i < data._cells.size(); ++i) {
        EXPECT_FLOAT_EQ(originalPositions[i].x, data._cells[i]._pos.x);
        EXPECT_FLOAT_EQ(originalPositions[i].y, data._cells[i]._pos.y);
    }
}

TEST_F(DescriptionEditServiceTests, flattenTopology_disconnectedComponents_withWrapping)
{
    // Two separate components, one of which needs topology correction
    Description data = createDisconnectedComponents(
        {{5.0f, 50.0f}, {95.0f, 50.0f}},   // Component 1: needs horizontal wrapping correction
        {{50.0f, 25.0f}, {50.0f, 27.0f}}   // Component 2: no correction needed
    );
    IntVector2D worldSize{100, 100};
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    ASSERT_EQ(4, data._cells.size());
    
    // Component 2 should be unchanged
    auto cell3Pos = data._cells[2]._pos;
    auto cell4Pos = data._cells[3]._pos;
    EXPECT_FLOAT_EQ(50.0f, cell3Pos.x);
    EXPECT_FLOAT_EQ(25.0f, cell3Pos.y);
    EXPECT_FLOAT_EQ(50.0f, cell4Pos.x);
    EXPECT_FLOAT_EQ(27.0f, cell4Pos.y);
    
    // Component 1 should have topology correction applied
    auto cell1Pos = data._cells[0]._pos;
    auto cell2Pos = data._cells[1]._pos;
    bool bothOnLeft = (cell1Pos.x < 50.0f && cell2Pos.x < 50.0f);
    bool bothOnRight = (cell1Pos.x > 50.0f && cell2Pos.x > 50.0f);
    EXPECT_TRUE(bothOnLeft || bothOnRight);
}

TEST_F(DescriptionEditServiceTests, flattenTopology_cellsWithoutConnections)
{
    // Cells without connections should not be affected
    Description data = Description().cells({
        CellDescription().id(1).pos({10.0f, 20.0f}),
        CellDescription().id(2).pos({90.0f, 80.0f}),
        CellDescription().id(3).pos({50.0f, 50.0f})
    });
    IntVector2D worldSize{100, 100};
    
    auto originalPositions = std::vector<RealVector2D>();
    for (const auto& cell : data._cells) {
        originalPositions.push_back(cell._pos);
    }
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    ASSERT_EQ(3, data._cells.size());
    // Positions should remain unchanged since there are no connections
    for (size_t i = 0; i < data._cells.size(); ++i) {
        EXPECT_FLOAT_EQ(originalPositions[i].x, data._cells[i]._pos.x);
        EXPECT_FLOAT_EQ(originalPositions[i].y, data._cells[i]._pos.y);
    }
}

TEST_F(DescriptionEditServiceTests, flattenTopology_ring_topology)
{
    // Create a ring: cell1 -> cell2 -> cell3 -> cell1, where cell3 wraps around
    std::vector<CellDescription> cells;
    cells.push_back(CellDescription().id(1).pos({10.0f, 50.0f}).connections({
        ConnectionDescription().cellId(2), ConnectionDescription().cellId(3)
    }));
    cells.push_back(CellDescription().id(2).pos({20.0f, 50.0f}).connections({
        ConnectionDescription().cellId(1), ConnectionDescription().cellId(3)
    }));
    cells.push_back(CellDescription().id(3).pos({95.0f, 50.0f}).connections({  // This one wraps
        ConnectionDescription().cellId(1), ConnectionDescription().cellId(2)
    }));
    
    Description data = Description().cells(cells);
    IntVector2D worldSize{100, 100};
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    ASSERT_EQ(3, data._cells.size());
    
    // After flattening, cell3 should be corrected to be near the other cells
    // Find positions after correction
    std::map<uint64_t, RealVector2D> posById;
    for (const auto& cell : data._cells) {
        posById[cell._id] = cell._pos;
    }
    
    // All cells should be on the same "side" of the world
    float maxX = std::max({posById[1].x, posById[2].x, posById[3].x});
    float minX = std::min({posById[1].x, posById[2].x, posById[3].x});
    
    // The range should be reasonable (much less than world size)
    EXPECT_LT(maxX - minX, 50.0f);
}

TEST_F(DescriptionEditServiceTests, flattenTopology_complex_wrapping_both_axes)
{
    // Test wrapping in both X and Y dimensions
    Description data = createConnectedPair({5.0f, 5.0f}, {95.0f, 95.0f});
    IntVector2D worldSize{100, 100};
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    ASSERT_EQ(2, data._cells.size());
    
    auto pos1 = data._cells[0]._pos;
    auto pos2 = data._cells[1]._pos;
    
    // After correction, cells should be close in both dimensions
    float xDist = std::abs(pos1.x - pos2.x);
    float yDist = std::abs(pos1.y - pos2.y);
    
    // Distances should be much smaller than half the world size
    EXPECT_LT(xDist, 50.0f);
    EXPECT_LT(yDist, 50.0f);
}

TEST_F(DescriptionEditServiceTests, flattenTopology_mixed_connected_and_unconnected)
{
    // Mix of connected and unconnected cells
    Description data;
    data._cells = {
        // Connected pair that needs correction
        CellDescription().id(1).pos({5.0f, 50.0f}).connections({ConnectionDescription().cellId(2)}),
        CellDescription().id(2).pos({95.0f, 50.0f}).connections({ConnectionDescription().cellId(1)}),
        // Isolated cells that should not change
        CellDescription().id(3).pos({30.0f, 30.0f}),
        CellDescription().id(4).pos({70.0f, 70.0f})
    };
    IntVector2D worldSize{100, 100};
    
    auto isolatedPos3 = data._cells[2]._pos;
    auto isolatedPos4 = data._cells[3]._pos;
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    ASSERT_EQ(4, data._cells.size());
    
    // Find cells by ID after processing
    std::map<uint64_t, RealVector2D> posById;
    for (const auto& cell : data._cells) {
        posById[cell._id] = cell._pos;
    }
    
    // Isolated cells should not have moved
    EXPECT_FLOAT_EQ(isolatedPos3.x, posById[3].x);
    EXPECT_FLOAT_EQ(isolatedPos3.y, posById[3].y);
    EXPECT_FLOAT_EQ(isolatedPos4.x, posById[4].x);
    EXPECT_FLOAT_EQ(isolatedPos4.y, posById[4].y);
    
    // Connected cells should be corrected to be on the same side
    bool bothOnLeft = (posById[1].x < 50.0f && posById[2].x < 50.0f);
    bool bothOnRight = (posById[1].x > 50.0f && posById[2].x > 50.0f);
    EXPECT_TRUE(bothOnLeft || bothOnRight);
}

TEST_F(DescriptionEditServiceTests, flattenTopology_cells_in_creatures)
{
    // Test topology correction for cells within creatures
    Description data = Description().creatures({
        CreatureDescription().id(1).cells({
            CellDescription().id(10).pos({5.0f, 50.0f}).connections({ConnectionDescription().cellId(11)}),
            CellDescription().id(11).pos({95.0f, 50.0f}).connections({ConnectionDescription().cellId(10)})
        })
    });
    IntVector2D worldSize{100, 100};
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    ASSERT_EQ(1, data._creatures.size());
    ASSERT_EQ(2, data._creatures[0]._cells.size());
    
    auto cell1Pos = data._creatures[0]._cells[0]._pos;
    auto cell2Pos = data._creatures[0]._cells[1]._pos;
    
    // After flattening, both cells should be on the same side
    bool bothOnLeft = (cell1Pos.x < 50.0f && cell2Pos.x < 50.0f);
    bool bothOnRight = (cell1Pos.x > 50.0f && cell2Pos.x > 50.0f);
    EXPECT_TRUE(bothOnLeft || bothOnRight);
}

TEST_F(DescriptionEditServiceTests, flattenTopology_mixed_individual_and_creature_cells)
{
    // Test with both individual cells and creature cells
    Description data;
    data._cells = {
        CellDescription().id(1).pos({10.0f, 50.0f}).connections({ConnectionDescription().cellId(2)}),
        CellDescription().id(2).pos({90.0f, 50.0f}).connections({ConnectionDescription().cellId(1)})
    };
    data._creatures = {
        CreatureDescription().id(10).cells({
            CellDescription().id(3).pos({15.0f, 25.0f}).connections({ConnectionDescription().cellId(4)}),
            CellDescription().id(4).pos({85.0f, 25.0f}).connections({ConnectionDescription().cellId(3)})
        })
    };
    IntVector2D worldSize{100, 100};
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    ASSERT_EQ(2, data._cells.size());
    ASSERT_EQ(1, data._creatures.size());
    ASSERT_EQ(2, data._creatures[0]._cells.size());
    
    // Check individual cells
    std::map<uint64_t, RealVector2D> cellPosById;
    for (const auto& cell : data._cells) {
        cellPosById[cell._id] = cell._pos;
    }
    bool individualCellsBothOnLeft = (cellPosById[1].x < 50.0f && cellPosById[2].x < 50.0f);
    bool individualCellsBothOnRight = (cellPosById[1].x > 50.0f && cellPosById[2].x > 50.0f);
    EXPECT_TRUE(individualCellsBothOnLeft || individualCellsBothOnRight);
    
    // Check creature cells
    std::map<uint64_t, RealVector2D> creatureCellPosById;
    for (const auto& cell : data._creatures[0]._cells) {
        creatureCellPosById[cell._id] = cell._pos;
    }
    bool creatureCellsBothOnLeft = (creatureCellPosById[3].x < 50.0f && creatureCellPosById[4].x < 50.0f);
    bool creatureCellsBothOnRight = (creatureCellPosById[3].x > 50.0f && creatureCellPosById[4].x > 50.0f);
    EXPECT_TRUE(creatureCellsBothOnLeft || creatureCellsBothOnRight);
}

TEST_F(DescriptionEditServiceTests, flattenTopology_manual_validation_simple_case)
{
    // Manual validation: ensure our understanding of the function is correct
    // Case: two cells connected, one at each side of a 10x10 world
    Description data = createConnectedPair({1.0f, 5.0f}, {9.0f, 5.0f});
    IntVector2D worldSize{10, 10};
    
    DescriptionEditService::get().flattenTopology(data, worldSize);
    
    ASSERT_EQ(2, data._cells.size());
    
    // One of the cells should have been moved to be near the other
    // In a 10x10 world, position 9 should be corrected to -1 (equivalent to 9 but on the left side)
    // or position 1 should be corrected to 11 (equivalent to 1 but on the right side)
    auto pos1 = data._cells[0]._pos;
    auto pos2 = data._cells[1]._pos;
    
    // The distance between the cells should be small after correction
    float distance = std::abs(pos1.x - pos2.x);
    EXPECT_LT(distance, 5.0f);  // Should be much less than half the world size
}