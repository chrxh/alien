#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class SensorTests : public IntegrationTestFramework
{
public:
    SensorTests()
        : IntegrationTestFramework()
    {}

    ~SensorTests() = default;

protected:
    // Helper to create a sensor mode description with appropriate minDensity
    // Note: Creates a fresh description with the specified density since test parameters use default modes
    static SensorModeDescription createModeWithDensity(SensorModeDescription const& mode, float minDensity)
    {
        if (std::holds_alternative<DetectEnergyDescription>(mode)) {
            return DetectEnergyDescription().minDensity(minDensity);
        } else if (std::holds_alternative<DetectFreeCellDescription>(mode)) {
            return DetectFreeCellDescription().minDensity(minDensity);
        } else {
            return mode;  // DetectStructureDescription has no minDensity
        }
    }

    // Helper to add detection targets for row layout (horizontal line of targets)
    void addDetectionTargetsRow(Description& data, SensorModeDescription const& mode, RealVector2D const& startPos, int count = 8)
    {
        if (std::holds_alternative<DetectEnergyDescription>(mode)) {
            for (int i = 0; i < count; ++i) {
                data._particles.emplace_back(ParticleDescription().pos({startPos.x + i, startPos.y}).energy(10.0f));
            }
        } else if (std::holds_alternative<DetectFreeCellDescription>(mode)) {
            for (int i = 0; i < count; ++i) {
                data._cells.emplace_back(
                    CellDescription().pos({startPos.x + i, startPos.y}).cellType(FreeCellDescription()));
            }
        } else if (std::holds_alternative<DetectStructureDescription>(mode)) {
            for (int i = 0; i < count; ++i) {
                data._cells.emplace_back(CellDescription().pos({startPos.x + i, startPos.y}).cellType(StructureCellDescription()));
            }
        }
    }
};

/**
 * Parameterized test class for sensor modes that share common behavior
 */
class SensorTests_AllModes
    : public SensorTests
    , public testing::WithParamInterface<SensorModeDescription>
{
protected:
    SensorModeDescription getMode() const { return GetParam(); }

    SensorModeDescription getModeWithDensity(float minDensity) const { return createModeWithDensity(GetParam(), minDensity); }
};

INSTANTIATE_TEST_SUITE_P(
    SensorModes,
    SensorTests_AllModes,
    ::testing::Values(DetectEnergyDescription(), DetectStructureDescription(), DetectFreeCellDescription()));

/**
 * Parameterized tests that work across all sensor modes
 */

TEST_P(SensorTests_AllModes, autoTriggered)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(15).mode(getMode())),
    });
    _simulationFacade->setSimulationData(data);

    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_TRUE(actualSensor._signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_FALSE(actualSensor._signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(14);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_TRUE(actualSensor._signal.has_value());
    }
}

TEST_P(SensorTests_AllModes, manuallyTriggered_noSignal)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(std::nullopt).mode(getMode())),
    });
    _simulationFacade->setSimulationData(data);

    for (int i = 0; i < 10; ++i) {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_FALSE(actualSensor._signal.has_value());
    }
}

TEST_P(SensorTests_AllModes, manuallyTriggered_withSignal)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(std::nullopt).mode(getMode())),
        CellDescription().id(2).pos({101.0f, 100.0f}).signalAndState({1, 0, 0, 0, 0, 0, 0, 0}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
}

TEST_P(SensorTests_AllModes, targetAbove)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(getModeWithDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add detection targets above the sensor
    addDetectionTargetsRow(data, getMode(), {98.0f, 20.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    // DetectEnergy and DetectFreeCell modes return density, DetectStructure does not
    if (!std::holds_alternative<DetectStructureDescription>(getMode())) {
        EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.0f);
    }
    // Angle should be roughly -90 degrees (-0.5 normalized)
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < -0.3f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > -0.7f);
}

TEST_P(SensorTests_AllModes, targetBelow)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(getModeWithDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add detection targets below the sensor
    addDetectionTargetsRow(data, getMode(), {98.0f, 180.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    // DetectEnergy and DetectFreeCell modes return density, DetectStructure does not
    if (!std::holds_alternative<DetectStructureDescription>(getMode())) {
        EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.0f);
    }
    // Angle should be roughly +90 degrees (+0.5 normalized)
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > 0.3f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < 0.7f);
}

TEST_P(SensorTests_AllModes, closerTargetDetected)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(getModeWithDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add a close cluster
    addDetectionTargetsRow(data, getMode(), {98.0f, 50.0f}, 8);

    // Add a far cluster with more items
    addDetectionTargetsRow(data, getMode(), {98.0f, 200.0f}, 12);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    // Should detect the closer targets (above the sensor)
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 0.6f);  // Closer = higher value
}

TEST_P(SensorTests_AllModes, minRange_found)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(getModeWithDensity(0.05f)).minRange(40)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add targets just beyond minRange
    addDetectionTargetsRow(data, getMode(), {98.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllModes, minRange_notFound)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(getModeWithDensity(0.05f)).minRange(120)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add targets within minRange (too close)
    addDetectionTargetsRow(data, getMode(), {98.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllModes, maxRange_found)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(getModeWithDensity(0.05f)).maxRange(120)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add targets within maxRange
    addDetectionTargetsRow(data, getMode(), {98.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllModes, maxRange_notFound)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(getModeWithDensity(0.05f)).maxRange(30)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add targets beyond maxRange (too far)
    addDetectionTargetsRow(data, getMode(), {98.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllModes, rayBlockedBySameCreatureConnections)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(getModeWithDensity(0.05f))),

        // Create a connection that crosses the ray path
        CellDescription().id(2).pos({99.0f, 99.0f}),
        CellDescription().id(3).pos({101.0f, 99.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(1, 3);

    // Add detection targets above the sensor
    addDetectionTargetsRow(data, getMode(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    // Ray should be blocked by same-creature connection
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllModes, rayNotBlockedByDifferentCreature)
{
    auto data = Description()
                    .addCreature(CreatureDescription().id(0).cells(
                        {CellDescription()
                             .id(1)
                             .pos({100.0f, 100.0f})
                             .frontAngle(0.0f)
                             .cellType(SensorDescription().autoTriggerInterval(3).mode(getModeWithDensity(0.05f))),
                         CellDescription().id(2).pos({101.0f, 100.0f})}))
                    .addCreature(CreatureDescription().cells({
                        // Create a different creature with a connection that would cross the ray path
                        CellDescription().id(10).pos({100.0f, 99.0f}),
                        CellDescription().id(11).pos({101.0f, 99.0f}),
                    }));
    data.addConnection(1, 2);    // Sensor creature
    data.addConnection(10, 11);  // Different creature

    // Add detection targets above the sensor
    addDetectionTargetsRow(data, getMode(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    // Ray should NOT be blocked by different creature connections
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

/**
 * Non-parameterized tests for mode-specific behavior
 */

TEST_F(SensorTests, detectEnergy_noFrontAngle)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.1f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    data._particles.emplace_back(ParticleDescription().id(100).pos({100.0f, 10.0f}).energy(50.0f));

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_FALSE(actualSensor._signal.has_value());
}

TEST_F(SensorTests, detectEnergy_particleFound)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.5f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add energy particles near the sensor - cluster them in same 8x8 region to accumulate energy
    for (int i = 0; i < 10; ++i) {
        data._particles.emplace_back(ParticleDescription()
            .id(100 + i)
            .pos({100.0f + (i % 3) * 2.0f, 50.0f + (i / 3) * 2.0f})
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.0f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 0.0f);
}

TEST_F(SensorTests, detectEnergy_particleNotFound_lowEnergy)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(5.0f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add a particle with low energy
    data._particles.emplace_back(ParticleDescription().id(100).pos({100.0f, 50.0f}).energy(1.0f));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectEnergy_particleAbove_differentFrontAngle)
{
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(90.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.5f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add particles above the sensor - cluster them to reach minDensity
    for (int i = 0; i < 8; ++i) {
        data._particles.emplace_back(ParticleDescription().id(100 + i).pos({98.0f + i, 20.0f}).energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.0f);
    // Angle should be roughly -180 degrees
    auto angleAsSignal = actualSensor._signal->_channels[Channels::SensorAngle];
    EXPECT_TRUE(angleAsSignal < -0.9f || angleAsSignal > 0.9f);
}

TEST_F(SensorTests, detectEnergy_noParticles)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.5f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectEnergy_multipleDirections)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.5f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add particles in front (right side, angle 0)
    for (int i = 0; i < 8; ++i) {
        data._particles.emplace_back(ParticleDescription()
            .id(100 + i)
            .pos({150.0f, 98.0f + i})
            .energy(10.0f));
    }
    
    // Add particles behind (left side, angle 180)
    for (int i = 0; i < 8; ++i) {
        data._particles.emplace_back(ParticleDescription()
            .id(200 + i)
            .pos({50.0f, 98.0f + i})
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    // Should detect the closer one (right side at distance ~50 is closer than left at ~50)
    // Actually both are same distance, so it should detect the first one encountered
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.0f);
}

/**
 * Tests for ray blocking (both DetectEnergy and DetectFreeCell modes)
 */

TEST_F(SensorTests, detectEnergy_rayBlockedByStructureCells)
{
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.5f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add structure cells between sensor and energy particles (to block the ray)
    for (int i = 0; i < 5; ++i) {
        data._cells.emplace_back(CellDescription().id(50 + i).pos({100.0f, 50.0f + i * 2.0f}).cellType(StructureCellDescription()));
    }

    // Add energy particles behind the structure cells
    for (int i = 0; i < 10; ++i) {
        data._particles.emplace_back(ParticleDescription().id(100 + i).pos({98.0f + i, 20.0f}).energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    // Should not find energy particles because ray is blocked by structure cells
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectEnergy_rayNotBlockedByStructureCells_differentAngle)
{
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.5f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add structure cells on the left side
    for (int i = 0; i < 5; ++i) {
        data._cells.emplace_back(CellDescription().id(50 + i).pos({80.0f, 100.0f + i * 2.0f}).cellType(StructureCellDescription()));
    }

    // Add energy particles on the right side (not blocked)
    for (int i = 0; i < 10; ++i) {
        data._particles.emplace_back(ParticleDescription().id(100 + i).pos({120.0f + i, 100.0f}).energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    // Should find energy particles because ray is not blocked (different direction)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.0f);
}

/**
 * Tests for SensorMode_DetectFreeCell (mode-specific tests)
 */

TEST_F(SensorTests, detectFreeCell_freeCellsFound)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add free cells near the sensor - cluster them in same 8x8 region
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({100.0f + (i % 3) * 2.0f, 50.0f + (i / 3) * 2.0f})
            .cellType(FreeCellDescription())
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.0f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 0.0f);
}

TEST_F(SensorTests, detectFreeCell_notFound_lowDensity)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.5f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add just a few free cells
    data._cells.emplace_back(CellDescription().id(100).pos({100.0f, 50.0f}).cellType(FreeCellDescription()).energy(10.0f));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectFreeCell_restrictToColor)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).color(0).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f).restrictToColor(1))),
        CellDescription().id(2).pos({101.0f, 100.0f}).color(0),
    });
    data.addConnection(1, 2);
    
    // Add free cells with wrong color (color 0)
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({98.0f + i, 80.0f})
            .color(0)
            .cellType(FreeCellDescription())
            .energy(10.0f));
    }
    
    // Add free cells with correct color (color 1) but fewer
    for (int i = 0; i < 8; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(200 + i)
            .pos({98.0f + i, 250.0f})
            .color(1)
            .cellType(FreeCellDescription())
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    // Should detect the color 1 cells, not the color 0 cells
    // Color 1 cells are farther (below), so distance should be lower
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] < 0.6f);
}

TEST_F(SensorTests, detectFreeCell_ignoreDifferentCellTypes)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add many non-free cells (should be ignored)
    for (int i = 0; i < 20; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({98.0f + (i % 4), 50.0f + (i / 4)})
            .cellType(BaseDescription())
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    // Should not find anything because only non-free cells are present
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectFreeCell_rayBlockedByStructureCells)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add structure cells between sensor and free cells (to block the ray)
    for (int i = 0; i < 5; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(50 + i)
            .pos({100.0f, 50.0f + i * 2.0f})
            .cellType(StructureCellDescription()));
    }
    
    // Add free cells behind the structure cells
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({98.0f + i, 20.0f})
            .cellType(FreeCellDescription())
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    // Should not find free cells because ray is blocked by structure cells
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectFreeCell_rayNotBlockedByStructureCells_differentAngle)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add structure cells on the left side
    for (int i = 0; i < 5; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(50 + i)
            .pos({80.0f, 100.0f + i * 2.0f})
            .cellType(StructureCellDescription()));
    }
    
    // Add free cells on the right side (not blocked)
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({120.0f + i, 100.0f})
            .cellType(FreeCellDescription())
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    // Should find free cells because ray is not blocked (different direction)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.0f);
}

/**
 * Tests for SensorMode_DetectStructure (mode-specific tests)
 */

TEST_F(SensorTests, detectStructure_structureCellsFound)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectStructureDescription())),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add structure cells near the sensor (creates a 3x4 grid pattern)
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({100.0f + (i % 3) * 2.0f, 50.0f + (i / 3) * 2.0f})
            .cellType(StructureCellDescription()));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    // Structure detection returns no density (density should be 0)
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorDensity]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 0.0f);
}

TEST_F(SensorTests, detectStructure_noStructureCells)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectStructureDescription())),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectStructure_ignoreDifferentCellTypes)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectStructureDescription())),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add many non-structure cells (should be ignored)
    for (int i = 0; i < 20; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({98.0f + (i % 4), 50.0f + (i / 4)})
            .cellType(BaseDescription())
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    // Should not find anything because only non-structure cells are present
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

// Old tests for other sensor modes - temporarily disabled pending implementation
#if 0

TEST_F(SensorTests, autoTriggered)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(15)),
    };
    _simulationFacade->setSimulationData(data);

    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_TRUE(actualSensor._signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_FALSE(actualSensor._signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(14);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_TRUE(actualSensor._signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_FALSE(actualSensor._signal.has_value());
    }
}

TEST_F(SensorTests, manuallyTriggered_noSignal)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(0)),
    };
    _simulationFacade->setSimulationData(data);

    for (int i = 0; i < 100; ++i) {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_FALSE(actualSensor._signal.has_value());
    }
}

TEST_F(SensorTests, manuallyTriggered_signal)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(0)),
        CellDescription().id(2).pos({101.0f, 100.0f}).signalAndState({1, 0, 0, 0, 0, 0, 0, 0}),
    };
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
}

TEST_F(SensorTests, noFrontAngle)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellType(SensorDescription().autoTriggerInterval(3).minDensity(0.1f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(2.5f)));

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_FALSE(actualSensor._signal.has_value());
}

TEST_F(SensorTests, aboveMinDensity)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.2f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > (-90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < (-90.0f + 15.0f) / 180);
}

TEST_F(SensorTests, belowMinDensity)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).minDensity(0.1f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(2.5f)));

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, targetAbove)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.2f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > (-90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < (-90.0f + 15.0f) / 180);
}

TEST_F(SensorTests, targetBelow)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 190.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.2f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > (90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < (90.0f + 15.0f) / 180);
}

TEST_F(SensorTests, targetConcealed)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 101.0f}),
        CellDescription().id(3).pos({101.0f, 99.0f}),
    };
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(2, 3);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({190.0f, 100.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, targetNotConcealed)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 101.0f}),
        CellDescription().id(3).pos({101.0f, 99.0f}),
    };
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(2, 3);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 190.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, foundMassWithMatchingDensity)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).minDensity(0.7f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(1.5f)));
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 200.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.7f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > (90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < (90.0f + 15.0f) / 180);
}

TEST_F(SensorTests, scanForOtherMutants_found)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).lineageId(6).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToOtherMutants)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addCreature(CreatureDescription().lineageId(7).cells({
        DescriptionEditService::get()
            .createRect(
                DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
            ._cells,
    }));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorDensity] > 0.3f);
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorDistance] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorDistance] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorAngle] > (-90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorAngle] < (-90.0f + 15.0f) / 180);
}

TEST_F(SensorTests, scanForOtherMutants_found_wallBehind)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).lineageId(6).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToOtherMutants)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addCreature(CreatureDescription().lineageId(7).cells({
        DescriptionEditService::get()
            .createRect(
                DescriptionEditService::CreateRectParameters().center({200.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
            ._cells,
    }));
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({220.0f, 100.0f}).width(1).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForOtherMutants_notFound_wallInBetween)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).lineageId(6).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToOtherMutants)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addCreature(CreatureDescription().lineageId(7).cells({
        DescriptionEditService::get()
            .createRect(
                DescriptionEditService::CreateRectParameters().center({200.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
            ._cells,
    }));
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({150.0f, 100.0f}).width(1).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForOtherMutants_notFound_sameLineageId)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).lineageId(7).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToOtherMutants)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addCreature(CreatureDescription().lineageId(7).cells({
        DescriptionEditService::get()
            .createRect(
                DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
            ._cells,
    }));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForOtherMutants_notFound_structure)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).lineageId(7).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToOtherMutants)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForOtherMutants_notFound_freeCell)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).lineageId(7).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToOtherMutants)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForSameMutants_found)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).lineageId(6).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToSameMutants)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addCreature(CreatureDescription().lineageId(6).cells({
        DescriptionEditService::get()
            .createRect(
                DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
            ._cells,
    }));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForSameMutants_notFound_otherLineageId)
{
    auto const MutantId = 6;
    for (int otherMutantId = 0; otherMutantId < 100; ++otherMutantId) {
        if (otherMutantId == MutantId) {
            continue;
        }

        Description data;
        data.addCreature(CreatureDescription().id(0).lineageId(MutantId).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .frontAngle(0.0f)
                .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToSameMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }));
        data.addCreature(CreatureDescription()
                             .lineageId(otherMutantId)
                             .cells({
                                 DescriptionEditService::get()
                                     .createRect(DescriptionEditService::CreateRectParameters()
                                                     .center({10.0f, 100.0f})
                                                     .width(16)
                                                     .height(16)
                                                     .cellDistance(0.5f)
                                                     .cellType(BaseDescription()))
                                     ._cells,
                             }));
        data.addConnection(1, 2);

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = actualData.getCellRef(1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests, scanForSameMutants_notFound_structure)
{
    auto const MutantId = 6;

    Description data;
    data.addCreature(CreatureDescription().id(0).lineageId(MutantId).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToSameMutants)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->clear();
    _simulationFacade->setCurrentTimestep(0ull);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForSameMutants_notFound_freeCell)
{
    auto const MutantId = 6;

    Description data;
    data.addCreature(CreatureDescription().id(0).lineageId(MutantId).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToSameMutants)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->clear();
    _simulationFacade->setCurrentTimestep(0ull);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForStructures_found)
{
    Description data;
    data._cells = {
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToStructures)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForStructures_notFound)
{
    Description data;
    data._cells = {
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToStructures)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForFreeCells_found)
{
    Description data;
    data._cells = {
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToFreeCells)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForFreeCells_notFound)
{
    Description data;
    data._cells = {
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToFreeCells)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForLessComplexMutants_found)
{
    for (int otherNumCells = 0; otherNumCells < 500; ++otherNumCells) {
        Description data;
        data.addCreature(CreatureDescription().id(0).numCells(1000.0f).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .frontAngle(0.0f)
                .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToLessComplexMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }));
        data.addCreature(CreatureDescription()
                             .numCells(toFloat(otherNumCells))
                             .cells({
                                 DescriptionEditService::get()
                                     .createRect(DescriptionEditService::CreateRectParameters()
                                                     .center({10.0f, 100.0f})
                                                     .width(16)
                                                     .height(16)
                                                     .cellDistance(0.5f)
                                                     .cellType(BaseDescription()))
                                     ._cells,
                             }));
        data.addConnection(1, 2);

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = actualData.getCellRef(1);

        EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests, scanForLessComplexMutants_notFound_otherMoreComplex)
{
    for (int otherNumCells = 1000; otherNumCells < 2001; ++otherNumCells) {

        Description data;
        data.addCreature(CreatureDescription().id(0).numCells(1000.0f).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .frontAngle(0.0f)
                .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToLessComplexMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }));
        data.addCreature(CreatureDescription()
                             .numCells(toFloat(otherNumCells) + 0.01f)
                             .cells({
                                 DescriptionEditService::get()
                                     .createRect(DescriptionEditService::CreateRectParameters()
                                                     .center({10.0f, 100.0f})
                                                     .width(16)
                                                     .height(16)
                                                     .cellDistance(0.5f)
                                                     .cellType(BaseDescription()))
                                     ._cells,
                             }));
        data.addConnection(1, 2);

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = actualData.getCellRef(1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests, scanForLessComplexMutants_notFound_structure)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).numCells(1000.0f).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToLessComplexMutants)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForLessComplexMutants_notFound_freeCell)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).numCells(1000.0f).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToLessComplexMutants)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForMoreComplexMutants_found)
{
    for (int otherNumCells = 1000; otherNumCells < 2001; ++otherNumCells) {
        Description data;
        data.addCreature(CreatureDescription().id(0).numCells(500.0f).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .frontAngle(0.0f)
                .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToMoreComplexMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }));
        data.addCreature(CreatureDescription()
                             .numCells(toFloat(otherNumCells))
                             .cells({
                                 DescriptionEditService::get()
                                     .createRect(DescriptionEditService::CreateRectParameters()
                                                     .center({10.0f, 100.0f})
                                                     .width(16)
                                                     .height(16)
                                                     .cellDistance(0.5f)
                                                     .cellType(BaseDescription()))
                                     ._cells,
                             }));
        data.addConnection(1, 2);

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = actualData.getCellRef(1);

        EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests, scanForMoreComplexMutants_notFound_otherLessComplex)
{
    for (int otherNumCells = 0; otherNumCells < 500; ++otherNumCells) {
        Description data;
        data.addCreature(CreatureDescription().id(0).numCells(500.0f).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .frontAngle(0.0f)
                .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToMoreComplexMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }));
        data.addCreature(CreatureDescription()
                             .numCells(toFloat(otherNumCells))
                             .cells({
                                 DescriptionEditService::get()
                                     .createRect(DescriptionEditService::CreateRectParameters()
                                                     .center({10.0f, 100.0f})
                                                     .width(16)
                                                     .height(16)
                                                     .cellDistance(0.5f)
                                                     .cellType(BaseDescription()))
                                     ._cells,
                             }));
        data.addConnection(1, 2);


        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = actualData.getCellRef(1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests, scanForMoreComplexMutants_notFound_structure)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).numCells(100.0f).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToMoreComplexMutants)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForMoreComplexMutants_notFound_freeCell)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).numCells(100.0f).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToMoreComplexMutants)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, minRange_found)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().minRange(50)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, minRange_notFound)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().minRange(120)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, maxRange_found)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().maxRange(120)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, maxRange_notFound)
{
    Description data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().maxRange(50)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

#endif // Sensor tests temporarily disabled
