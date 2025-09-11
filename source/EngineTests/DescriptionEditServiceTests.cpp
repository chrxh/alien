#include <gtest/gtest.h>

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