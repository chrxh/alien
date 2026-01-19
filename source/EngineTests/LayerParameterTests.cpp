#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

/**
 * Tests for zone/layer-specific simulation parameters.
 * These tests verify that layer-dependent parameters (such as minCellEnergy)
 * are correctly applied to cells based on their position within different zones.
 * Tests cover:
 * - Circular zones with different radii
 * - Rectangular zones with different dimensions
 * - Overlapping zones
 * - Fadeout radius behavior
 */
class LayerParameterTests : public IntegrationTestFramework
{
public:
    LayerParameterTests()
        : IntegrationTestFramework({1000, 1000})
    {}

    ~LayerParameterTests() = default;

protected:
    /**
     * Helper function to create a cell at a specific position with given energy and color
     */
    ObjectDesc createCell(RealVector2D const& pos, float energy, int color = 0)
    {
        return ObjectDesc().pos(pos).color(color).vel({0, 0}).type(CellDesc().usableEnergy(energy));
    }

    /**
     * Helper function to setup a single circular layer
     */
    void setupCircularLayer(int layerIndex, RealVector2D const& center, float coreRadius, float fadeoutRadius)
    {
        _parameters.numLayers = layerIndex + 1;
        _parameters.layerOrderNumbers[layerIndex] = layerIndex + 1;
        _parameters.layerShape.layerValues[layerIndex] = LayerShapeType_Circular;
        _parameters.layerPosition.layerValues[layerIndex] = center;
        _parameters.layerCoreRadius.layerValues[layerIndex] = coreRadius;
        _parameters.layerFadeoutRadius.layerValues[layerIndex] = fadeoutRadius;
    }

    /**
     * Helper function to setup a single rectangular layer
     */
    void setupRectangularLayer(int layerIndex, RealVector2D const& center, RealVector2D const& rectSize, float fadeoutRadius)
    {
        _parameters.numLayers = layerIndex + 1;
        _parameters.layerOrderNumbers[layerIndex] = layerIndex + 1;
        _parameters.layerShape.layerValues[layerIndex] = LayerShapeType_Rectangular;
        _parameters.layerPosition.layerValues[layerIndex] = center;
        _parameters.layerCoreRect.layerValues[layerIndex] = rectSize;
        _parameters.layerFadeoutRadius.layerValues[layerIndex] = fadeoutRadius;
    }
};

/**
 * Test: Single circular zone with minCellEnergy
 * Verifies that cells inside a circular zone die when below the zone-specific minCellEnergy,
 * while cells outside the zone follow the base minCellEnergy value.
 */
TEST_F(LayerParameterTests, circularZone_minCellEnergy_cellsDieInsideZone)
{
    // Setup: Create a circular zone at center with higher minCellEnergy than base
    setupCircularLayer(0, {500.0f, 500.0f}, 100.0f, 50.0f);

    _parameters.minCellEnergy.baseValue[0] = 50.0f;              // Base value
    _parameters.minCellEnergy.layerValues[0].value[0] = 100.0f;  // Layer value
    _parameters.minCellEnergy.layerValues[0].enabled = true;

    _parameters.cellDeathProbability.baseValue[0] = 1.0f;  // High death probability for testing
    _parameters.radiationAbsorption.baseValue[0] = 0;      // Disable radiation

    _simulationFacade->setSimulationParameters(_parameters);

    // Create test cells:
    // - Cell inside zone with energy below layer minCellEnergy (should die)
    // - Cell inside zone with energy above layer minCellEnergy (should survive)
    // - Cell outside zone with energy below base minCellEnergy (should die)
    // - Cell outside zone with energy above base minCellEnergy (should survive)
    auto data = Desc().addCreature({
        createCell({500.0f, 500.0f}, 75.0f, 0),   // Inside zone, energy < layer min (should die)
        createCell({510.0f, 500.0f}, 120.0f, 0),  // Inside zone, energy > layer min (should survive)
        createCell({700.0f, 500.0f}, 40.0f, 0),   // Outside zone, energy < base min (should die)
        createCell({710.0f, 500.0f}, 60.0f, 0),   // Outside zone, energy > base min (should survive)
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(10);

    auto resultData = _simulationFacade->getSimulationData();

    // Verify results: expect 2 surviving cells
    EXPECT_EQ(2, resultData._objects.size());

    // The surviving cells should be the ones with sufficient energy for their zones
    bool foundInsideSurvivor = false;
    bool foundOutsideSurvivor = false;

    for (auto const& object : resultData._objects) {
        // Check if cell is near the expected positions
        if (approxCompare(object._pos.x, 510.0f, 10.0f) && approxCompare(object._pos.y, 500.0f, 10.0f)) {
            foundInsideSurvivor = true;
        }
        if (approxCompare(object._pos.x, 710.0f, 10.0f) && approxCompare(object._pos.y, 500.0f, 10.0f)) {
            foundOutsideSurvivor = true;
        }
    }

    EXPECT_TRUE(foundInsideSurvivor);
    EXPECT_TRUE(foundOutsideSurvivor);
}

/**
 * Test: Rectangular zone with minCellEnergy
 * Verifies that rectangular zones apply parameter values correctly based on position.
 */
TEST_F(LayerParameterTests, rectangularZone_minCellEnergy_cellsDieInsideZone)
{
    // Setup: Create a rectangular zone with higher minCellEnergy than base
    setupRectangularLayer(0, {500.0f, 500.0f}, {100.0f, 80.0f}, 10.0f);

    _parameters.minCellEnergy.baseValue[0] = 50.0f;              // Base value
    _parameters.minCellEnergy.layerValues[0].value[0] = 100.0f;  // Layer value
    _parameters.minCellEnergy.layerValues[0].enabled = true;

    _parameters.cellDeathProbability.baseValue[0] = 1.0f;
    _parameters.radiationAbsorption.baseValue[0] = 0;

    _simulationFacade->setSimulationParameters(_parameters);

    // Create test cells with energy well below minCellEnergy thresholds:
    // - Inside rectangle with energy = half of layer min (should die)
    // - Inside rectangle with high energy (should survive)
    // - Outside rectangle with energy = half of base min (should die)
    // - Outside rectangle with high energy (should survive)
    auto data = Desc().addCreature({
        createCell({500.0f, 500.0f}, 50.0f, 0),   // Inside rect, energy = 0.5 * layer min (should die)
        createCell({520.0f, 510.0f}, 120.0f, 0),  // Inside rect, energy > layer min (should survive)
        createCell({700.0f, 500.0f}, 25.0f, 0),   // Outside rect, energy = 0.5 * base min (should die)
        createCell({710.0f, 500.0f}, 70.0f, 0),   // Outside rect, energy > base min (should survive)
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(10);

    auto resultData = _simulationFacade->getSimulationData();

    // Verify results: expect 2 surviving cells
    EXPECT_EQ(2, resultData._objects.size());
}

/**
 * Test: Multiple circular zones with different sizes
 * Tests that zones with different core radii correctly apply their parameter values.
 */
TEST_F(LayerParameterTests, multipleCircularZones_differentSizes)
{
    // Setup: Create two circular zones with different sizes and minCellEnergy values
    _parameters.numLayers = 2;

    // Zone 1: Small zone at (300, 500) with radius 50
    _parameters.layerOrderNumbers[0] = 1;
    _parameters.layerShape.layerValues[0] = LayerShapeType_Circular;
    _parameters.layerPosition.layerValues[0] = {300.0f, 500.0f};
    _parameters.layerCoreRadius.layerValues[0] = 50.0f;
    _parameters.layerFadeoutRadius.layerValues[0] = 0.0f;
    _parameters.minCellEnergy.layerValues[0].value[0] = 100.0f;
    _parameters.minCellEnergy.layerValues[0].enabled = true;

    // Zone 2: Large zone at (700, 500) with radius 150
    _parameters.layerOrderNumbers[1] = 2;
    _parameters.layerShape.layerValues[1] = LayerShapeType_Circular;
    _parameters.layerPosition.layerValues[1] = {700.0f, 500.0f};
    _parameters.layerCoreRadius.layerValues[1] = 150.0f;
    _parameters.layerFadeoutRadius.layerValues[1] = 0.0f;
    _parameters.minCellEnergy.layerValues[1].value[0] = 120.0f;
    _parameters.minCellEnergy.layerValues[1].enabled = true;

    _parameters.minCellEnergy.baseValue[0] = 50.0f;
    _parameters.cellDeathProbability.baseValue[0] = 1.0f;
    _parameters.radiationAbsorption.baseValue[0] = 0;

    _simulationFacade->setSimulationParameters(_parameters);

    // Create test cells at different positions with very low energy for cells that should die
    auto data = Desc().addCreature({
        createCell({300.0f, 500.0f}, 10.0f, 0),   // Zone 1 center, very low energy (should die)
        createCell({305.0f, 500.0f}, 110.0f, 0),  // Zone 1, energy > 100 (should survive)
        createCell({700.0f, 500.0f}, 10.0f, 0),   // Zone 2 center, very low energy (should die)
        createCell({710.0f, 500.0f}, 130.0f, 0),  // Zone 2, energy > 120 (should survive)
        createCell({900.0f, 500.0f}, 10.0f, 0),   // Outside both zones, very low energy (should die)
        createCell({910.0f, 500.0f}, 70.0f, 0),   // Outside both zones, energy > 50 (should survive)
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(10);

    auto resultData = _simulationFacade->getSimulationData();

    // Verify: 3 cells should survive
    EXPECT_EQ(3, resultData._objects.size());
}

/**
 * Test: Overlapping circular zones
 * Tests behavior when zones overlap - the first applicable zone should take precedence.
 */
TEST_F(LayerParameterTests, overlappingCircularZones_parameterPrecedence)
{
    // Setup: Create two overlapping circular zones
    _parameters.numLayers = 2;

    // Zone 1: Lower order, at (500, 500), radius 100
    _parameters.layerOrderNumbers[0] = 1;
    _parameters.layerShape.layerValues[0] = LayerShapeType_Circular;
    _parameters.layerPosition.layerValues[0] = {500.0f, 500.0f};
    _parameters.layerCoreRadius.layerValues[0] = 100.0f;
    _parameters.layerFadeoutRadius.layerValues[0] = 0.0f;
    _parameters.minCellEnergy.layerValues[0].value[0] = 100.0f;
    _parameters.minCellEnergy.layerValues[0].enabled = true;

    // Zone 2: Higher order, at (550, 500), radius 80 (overlaps with Zone 1)
    _parameters.layerOrderNumbers[1] = 2;
    _parameters.layerShape.layerValues[1] = LayerShapeType_Circular;
    _parameters.layerPosition.layerValues[1] = {550.0f, 500.0f};
    _parameters.layerCoreRadius.layerValues[1] = 80.0f;
    _parameters.layerFadeoutRadius.layerValues[1] = 0.0f;
    _parameters.minCellEnergy.layerValues[1].value[0] = 120.0f;
    _parameters.minCellEnergy.layerValues[1].enabled = true;

    _parameters.minCellEnergy.baseValue[0] = 50.0f;
    _parameters.cellDeathProbability.baseValue[0] = 1.0f;
    _parameters.radiationAbsorption.baseValue[0] = 0;

    _simulationFacade->setSimulationParameters(_parameters);

    // Create cells in different regions with very low energy for cells that should die:
    // - Only in Zone 1
    // - In overlapping region (both zones)
    // - Only in Zone 2
    auto data = Desc().addCreature({
        createCell({450.0f, 500.0f}, 10.0f, 0),   // Only Zone 1, very low energy (should die)
        createCell({460.0f, 500.0f}, 110.0f, 0),  // Only Zone 1, energy > 100 (should survive)
        createCell({525.0f, 500.0f}, 10.0f, 0),   // Overlap, very low energy (should die)
        createCell({530.0f, 500.0f}, 130.0f, 0),  // Overlap, energy > 120 (should survive)
        createCell({610.0f, 500.0f}, 10.0f, 0),   // Only Zone 2, very low energy (should die)
        createCell({620.0f, 500.0f}, 140.0f, 0),  // Only Zone 2, energy > 120 (should survive)
        createCell({300.0f, 500.0f}, 10.0f, 0),   // Outside zones, very low energy (should die)
        createCell({310.0f, 500.0f}, 70.0f, 0),   // Outside zones, energy > 50 (should survive)
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(10);

    auto resultData = _simulationFacade->getSimulationData();

    // Verify: 4 cells should survive (one from each region: zone1 only, overlap, zone2 only, outside)
    EXPECT_EQ(4, resultData._objects.size());
}

/**
 * Test: Circular zone with fadeout radius
 * Tests that the fadeout radius creates a transition region where parameter values are interpolated.
 */
TEST_F(LayerParameterTests, circularZone_fadeoutRadius_transitionBehavior)
{
    // Setup: Create a zone with significant fadeout radius
    setupCircularLayer(0, {500.0f, 500.0f}, 80.0f, 60.0f);

    _parameters.minCellEnergy.baseValue[0] = 50.0f;
    _parameters.minCellEnergy.layerValues[0].value[0] = 150.0f;
    _parameters.minCellEnergy.layerValues[0].enabled = true;

    _parameters.cellDeathProbability.baseValue[0] = 1.0f;
    _parameters.radiationAbsorption.baseValue[0] = 0;

    _simulationFacade->setSimulationParameters(_parameters);

    // Create cells at different distances from center to test fadeout behavior
    // Core radius: 80, fadeout: 60, so full transition ends at distance 140
    auto data = Desc().addCreature({
        createCell({500.0f, 500.0f}, 160.0f, 0),  // Center, full layer effect, energy > 150 (should survive)
        createCell({570.0f, 500.0f}, 75.0f, 0),   // In fadeout zone (distance ~70), energy = 0.5 * 150 (should die)
        createCell({650.0f, 500.0f}, 60.0f, 0),   // Outside fadeout (distance ~150), energy > 50 base (should survive)
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(10);

    auto resultData = _simulationFacade->getSimulationData();

    // Cell at center should survive (energy > layer min)
    // Cell outside fadeout should survive (energy > base min)
    // At least 2 cells should survive
    EXPECT_GE(resultData._objects.size(), 2);
}

/**
 * Test: Rectangular zone dimensions
 * Tests that rectangular zones correctly use their width and height dimensions.
 */
TEST_F(LayerParameterTests, rectangularZone_dimensions_correctBoundaries)
{
    // Setup: Create a wide but short rectangular zone
    setupRectangularLayer(0, {500.0f, 500.0f}, {200.0f, 50.0f}, 10.0f);

    _parameters.minCellEnergy.baseValue[0] = 50.0f;
    _parameters.minCellEnergy.layerValues[0].value[0] = 100.0f;
    _parameters.minCellEnergy.layerValues[0].enabled = true;

    _parameters.cellDeathProbability.baseValue[0] = 1.0f;
    _parameters.radiationAbsorption.baseValue[0] = 0;

    _simulationFacade->setSimulationParameters(_parameters);

    // Create cells to test horizontal and vertical boundaries
    auto data = Desc().addCreature({
        createCell({500.0f, 500.0f}, 110.0f, 0),  // Center, inside (survives)
        createCell({580.0f, 500.0f}, 110.0f, 0),  // Inside horizontally (survives)
        createCell({500.0f, 520.0f}, 50.0f, 0),   // Inside vertically, energy = 0.5 * 100 (dies)
        createCell({500.0f, 560.0f}, 60.0f, 0),   // Outside vertically, energy > 50 (survives)
        createCell({710.0f, 500.0f}, 25.0f, 0),   // Outside horizontally, energy = 0.5 * 50 (dies)
        createCell({720.0f, 500.0f}, 60.0f, 0),   // Outside horizontally, energy > 50 (survives)
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(10);

    auto resultData = _simulationFacade->getSimulationData();

    // Expect 4 survivors
    EXPECT_EQ(4, resultData._objects.size());
}

/**
 * Test: Mixed zone shapes
 * Tests that circular and rectangular zones can coexist and work correctly.
 */
TEST_F(LayerParameterTests, mixedZoneShapes_circularAndRectangular)
{
    // Setup: Create one circular and one rectangular zone
    _parameters.numLayers = 2;

    // Zone 1: Circular at (300, 500)
    _parameters.layerOrderNumbers[0] = 1;
    _parameters.layerShape.layerValues[0] = LayerShapeType_Circular;
    _parameters.layerPosition.layerValues[0] = {300.0f, 500.0f};
    _parameters.layerCoreRadius.layerValues[0] = 60.0f;
    _parameters.layerFadeoutRadius.layerValues[0] = 0.0f;
    _parameters.minCellEnergy.layerValues[0].value[0] = 100.0f;
    _parameters.minCellEnergy.layerValues[0].enabled = true;

    // Zone 2: Rectangular at (700, 500)
    _parameters.layerOrderNumbers[1] = 2;
    _parameters.layerShape.layerValues[1] = LayerShapeType_Rectangular;
    _parameters.layerPosition.layerValues[1] = {700.0f, 500.0f};
    _parameters.layerCoreRect.layerValues[1] = {100.0f, 80.0f};
    _parameters.layerFadeoutRadius.layerValues[1] = 0.0f;
    _parameters.minCellEnergy.layerValues[1].value[0] = 110.0f;
    _parameters.minCellEnergy.layerValues[1].enabled = true;

    _parameters.minCellEnergy.baseValue[0] = 50.0f;
    _parameters.cellDeathProbability.baseValue[0] = 1.0f;
    _parameters.radiationAbsorption.baseValue[0] = 0;

    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Desc().addCreature({
        createCell({300.0f, 500.0f}, 110.0f, 0),  // Circular zone, energy > 100 (survives)
        createCell({310.0f, 500.0f}, 10.0f, 0),   // Circular zone, very low energy (dies)
        createCell({700.0f, 500.0f}, 120.0f, 0),  // Rectangular zone, energy > 110 (survives)
        createCell({710.0f, 500.0f}, 10.0f, 0),   // Rectangular zone, very low energy (dies)
        createCell({900.0f, 500.0f}, 10.0f, 0),   // Outside both, very low energy (dies)
        createCell({910.0f, 500.0f}, 70.0f, 0),   // Outside both, energy > 50 (survives)
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(10);

    auto resultData = _simulationFacade->getSimulationData();

    // Expect 3 survivors
    EXPECT_EQ(3, resultData._objects.size());
}

/**
 * Test: Moving rectangular zone
 * Tests that a rectangular zone with velocity correctly applies its parameter values
 * to cells as the zone moves through space.
 */
TEST_F(LayerParameterTests, movingRectangularZone_cellsAffectedByMovingZone)
{
    // Setup: Create a rectangular zone that moves horizontally with velocity
    setupRectangularLayer(0, {300.0f, 500.0f}, {100.0f, 80.0f}, 20.0f);

    // Set zone velocity - moving rightward at 1.0 unit per timestep
    _parameters.layerVelocity.layerValues[0] = {1.0f, 0.0f};

    _parameters.minCellEnergy.baseValue[0] = 50.0f;
    _parameters.minCellEnergy.layerValues[0].value[0] = 120.0f;
    _parameters.minCellEnergy.layerValues[0].enabled = true;

    _parameters.cellDeathProbability.baseValue[0] = 1.0f;
    _parameters.radiationAbsorption.baseValue[0] = 0;

    _simulationFacade->setSimulationParameters(_parameters);

    // Create stationary cells at different positions:
    // - Cell at zone's initial position (should die immediately)
    // - Cell ahead of zone's path (initially outside, will be inside after zone moves)
    // - Cell far from zone's path (should survive throughout)
    auto data = Desc().addCreature({
        createCell({300.0f, 500.0f}, 110.0f, 0),  // At initial zone center, dies (< 120)
        createCell({320.0f, 500.0f}, 125.0f, 0),  // Inside initial zone, survives (> 120)
        createCell({450.0f, 500.0f}, 110.0f, 0),  // Ahead of zone, initially safe but will be caught
        createCell({460.0f, 500.0f}, 130.0f, 0),  // Ahead of zone, will survive when caught
        createCell({300.0f, 650.0f}, 55.0f, 0),   // Far outside zone path, survives (> 50 base)
    });

    _simulationFacade->setSimulationData(data);

    // Run simulation for 100 timesteps
    // Zone moves from x=300 to x=400 (100 units rightward)
    // This should bring the zone over the cells at x=450, x=460
    _simulationFacade->calcTimesteps(100);

    auto resultData = _simulationFacade->getSimulationData();

    // Expected survivors:
    // - Cell at 320 (survives in initial zone)
    // - Cell at 460 (survives when zone moves over it)
    // - Cell at 300,650 (outside zone path)
    // Dead cells:
    // - Cell at 300 (dies in initial zone)
    // - Cell at 450 (dies when zone moves over it)
    EXPECT_EQ(3, resultData._objects.size());

    // Verify at least one survivor is near the expected positions
    bool foundInitialZoneSurvivor = false;
    bool foundCaughtSurvivor = false;
    bool foundOutsideSurvivor = false;

    for (auto const& object : resultData._objects) {
        // Check for cell that was initially in zone (around x=320)
        if (approxCompare(object._pos.x, 320.0f, 20.0f) && approxCompare(object._pos.y, 500.0f, 20.0f)) {
            foundInitialZoneSurvivor = true;
        }
        // Check for cell that got caught by moving zone (around x=460)
        if (approxCompare(object._pos.x, 460.0f, 20.0f) && approxCompare(object._pos.y, 500.0f, 20.0f)) {
            foundCaughtSurvivor = true;
        }
        // Check for cell outside zone path (around y=650)
        if (approxCompare(object._pos.y, 650.0f, 20.0f)) {
            foundOutsideSurvivor = true;
        }
    }

    EXPECT_TRUE(foundInitialZoneSurvivor);
    EXPECT_TRUE(foundCaughtSurvivor);
    EXPECT_TRUE(foundOutsideSurvivor);
}
