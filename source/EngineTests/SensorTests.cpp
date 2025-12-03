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
                data._cells.emplace_back(CellDescription().pos({startPos.x + i, startPos.y}).cellType(FreeCellDescription()));
            }
        } else if (std::holds_alternative<DetectStructureDescription>(mode)) {
            for (int i = 0; i < count; ++i) {
                data._cells.emplace_back(CellDescription().pos({startPos.x + i, startPos.y}).cellType(StructureCellDescription()));
            }
        } else if (std::holds_alternative<DetectCreatureDescription>(mode)) {
            // Create a large creature (10x10 grid) at the specified position
            std::vector<CellDescription> creatureCells;
            for (int j = 0; j < 10; ++j) {
                for (int i = 0; i < 10; ++i) {
                    creatureCells.emplace_back(CellDescription().pos({startPos.x + i, startPos.y + j}));
                }
            }
            auto creature = CreatureDescription().cells(creatureCells);
            data.addCreature(creature);
        }
    }
    
    Description createLargeCreature()
    {
        Description result;
        std::vector<CellDescription> target;
        for (int j = 0; j < 10; ++j) {
            for (int i = 0; i < 10; ++i) {
                target.emplace_back(CellDescription().id(10 + i + j * 10).pos({95.0f + i, 80.0f + j}));
            }
        }
        result.addCreature(CreatureDescription().id(1).cells(target));
        for (int j = 0; j < 10; ++j) {
            for (int i = 0; i < 9; ++i) {
                result.addConnection(10 + i + 10 * j, 11 + i + 10 * j);
            }
        }
        return result;
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
    ::testing::Values(DetectEnergyDescription(), DetectStructureDescription(), DetectFreeCellDescription(), DetectCreatureDescription()));

/**
 * Parameterized tests that work across all sensor modes
 */

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
//
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
//
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
    // DetectEnergy and DetectFreeCell modes return density, DetectStructure and DetectCreature do not
    if (!std::holds_alternative<DetectStructureDescription>(getMode()) && !std::holds_alternative<DetectCreatureDescription>(getMode())) {
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
    // DetectEnergy and DetectFreeCell modes return density, DetectStructure and DetectCreature do not
    if (!std::holds_alternative<DetectStructureDescription>(getMode()) && !std::holds_alternative<DetectCreatureDescription>(getMode())) {
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
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(getModeWithDensity(0.05f)).minRange(40)),
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
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(getModeWithDensity(0.05f)).minRange(120)),
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
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(getModeWithDensity(0.05f)).maxRange(120)),
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
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(getModeWithDensity(0.05f)).maxRange(30)),
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
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(getModeWithDensity(0.05f))),

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
 * Tests for SensorMode_DetectEnergy (mode-specific tests)
 */

TEST_F(SensorTests, detectEnergy_noFrontAngle)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.1f))),
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
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.5f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add energy particles near the sensor - cluster them in same 8x8 region to accumulate energy
    for (int i = 0; i < 10; ++i) {
        data._particles.emplace_back(ParticleDescription().id(100 + i).pos({100.0f + (i % 3) * 2.0f, 50.0f + (i / 3) * 2.0f}).energy(10.0f));
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
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(5.0f))),
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
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.5f))),
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
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.5f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add particles in front (right side, angle 0)
    for (int i = 0; i < 8; ++i) {
        data._particles.emplace_back(ParticleDescription().id(100 + i).pos({150.0f, 98.0f + i}).energy(10.0f));
    }

    // Add particles behind (left side, angle 180)
    for (int i = 0; i < 8; ++i) {
        data._particles.emplace_back(ParticleDescription().id(200 + i).pos({50.0f, 98.0f + i}).energy(10.0f));
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
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(CellDescription().id(50 + i).pos({95.0f + i, 50.0f}).cellType(StructureCellDescription()));
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

TEST_F(SensorTests, detectEnergy_relocation_targetStationary)
{
    // First scan - target is detected and position stored
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.5f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add energy particles
    for (int i = 0; i < 10; ++i) {
        data._particles.emplace_back(ParticleDescription().id(100 + i).pos({100.0f + (i % 3) * 2.0f, 50.0f + (i / 3) * 2.0f}).energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Verify lastMatch was stored
    auto sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());

    // Second scan - target hasn't moved, should still be found via relocation
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Verify lastMatch is still present
    sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());

    // Note: During relocation, density is set to 0 (not recalculated)
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorDensity]));
}

TEST_F(SensorTests, detectEnergy_relocation_targetMoved)
{
    // First scan - target is detected and position stored
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.5f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add energy particles
    for (int i = 0; i < 10; ++i) {
        data._particles.emplace_back(ParticleDescription().id(100 + i).pos({100.0f + (i % 3) * 2.0f, 50.0f + (i / 3) * 2.0f}).energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Get the initial distance
    auto initialDistance = actualSensor._signal->_channels[Channels::SensorDistance];

    // Move the target particles
    actualData = _simulationFacade->getSimulationData();
    for (auto& particle : actualData._particles) {
        particle._pos.x += 10.0f;  // Move right by 10 units
    }
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target has moved, should be relocated via tracking
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Distance should be different since target moved
    auto newDistance = actualSensor._signal->_channels[Channels::SensorDistance];
    EXPECT_FALSE(approxCompare(initialDistance, newDistance));

    // Verify lastMatch position was updated
    auto sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());
}

TEST_F(SensorTests, detectEnergy_relocation_targetDisappeared)
{
    // First scan - target is detected and position stored
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.5f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add energy particles
    for (int i = 0; i < 10; ++i) {
        data._particles.emplace_back(ParticleDescription().id(100 + i).pos({100.0f + (i % 3) * 2.0f, 50.0f + (i / 3) * 2.0f}).energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Remove all particles
    actualData._particles.clear();
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target disappeared, should not be found
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Verify lastMatch was cleared
    auto sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

TEST_F(SensorTests, detectEnergy_relocation_targetBlocked)
{
    // First scan - target is detected and position stored
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(0.5f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add energy particles
    for (int i = 0; i < 10; ++i) {
        data._particles.emplace_back(ParticleDescription().id(100 + i).pos({100.0f + (i % 3) * 2.0f, 50.0f + (i / 3) * 2.0f}).energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Add structure cells between sensor and target to block the ray
    actualData = _simulationFacade->getSimulationData();
    for (int i = 0; i < 10; ++i) {
        actualData._cells.emplace_back(CellDescription().id(50 + i).pos({95.0f + i, 75.0f}).cellType(StructureCellDescription()));
    }
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target is now blocked by structure cells
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Verify lastMatch was cleared
    auto sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

/**
 * Tests for SensorMode_DetectStructure (mode-specific tests)
 */

TEST_F(SensorTests, detectStructure_structureCellsFound)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectStructureDescription())),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add structure cells near the sensor (creates a 3x4 grid pattern)
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(CellDescription().id(100 + i).pos({100.0f + (i % 3) * 2.0f, 50.0f + (i / 3) * 2.0f}).cellType(StructureCellDescription()));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorDensity]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 0.0f);
    EXPECT_FALSE(std::get<SensorDescription>(actualSensor._cellType)._lastMatch.has_value());
}

TEST_F(SensorTests, detectStructure_noStructureCells)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectStructureDescription())),
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
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectStructureDescription())),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add many non-structure cells (should be ignored)
    for (int i = 0; i < 20; ++i) {
        data._cells.emplace_back(CellDescription().id(100 + i).pos({98.0f + (i % 4), 50.0f + (i / 4)}).cellType(BaseDescription()).energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    // Should not find anything because only non-structure cells are present
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

/**
 * Tests for SensorMode_DetectFreeCell (mode-specific tests)
 */

TEST_F(SensorTests, detectFreeCell_freeCellsFound)
{
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add free cells near the sensor - cluster them in same 8x8 region
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(
            CellDescription().id(100 + i).pos({100.0f + (i % 3) * 2.0f, 50.0f + (i / 3) * 2.0f}).cellType(FreeCellDescription()).energy(10.0f));
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
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.5f))),
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
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .color(0)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f).restrictToColor(1))),
        CellDescription().id(2).pos({101.0f, 100.0f}).color(0),
    });
    data.addConnection(1, 2);

    // Add free cells with wrong color (color 0)
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(CellDescription().id(100 + i).pos({98.0f + i, 80.0f}).color(0).cellType(FreeCellDescription()).energy(10.0f));
    }

    // Add free cells with correct color (color 1) but fewer
    for (int i = 0; i < 8; ++i) {
        data._cells.emplace_back(CellDescription().id(200 + i).pos({98.0f + i, 250.0f}).color(1).cellType(FreeCellDescription()).energy(10.0f));
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
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add many non-free cells (should be ignored)
    for (int i = 0; i < 20; ++i) {
        data._cells.emplace_back(CellDescription().id(100 + i).pos({98.0f + (i % 4), 50.0f + (i / 4)}).cellType(BaseDescription()).energy(10.0f));
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
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add structure cells between sensor and free cells (to block the ray)
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(CellDescription().id(50 + i).pos({95.0f + i, 50.0f}).cellType(StructureCellDescription()));
    }

    // Add free cells behind the structure cells
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(CellDescription().id(100 + i).pos({98.0f + i, 20.0f}).cellType(FreeCellDescription()).energy(10.0f));
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
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add structure cells on the left side
    for (int i = 0; i < 5; ++i) {
        data._cells.emplace_back(CellDescription().id(50 + i).pos({80.0f, 100.0f + i * 2.0f}).cellType(StructureCellDescription()));
    }

    // Add free cells on the right side (not blocked)
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(CellDescription().id(100 + i).pos({120.0f + i, 100.0f}).cellType(FreeCellDescription()).energy(10.0f));
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

TEST_F(SensorTests, detectFreeCell_relocation_targetStationary)
{
    // First scan - target is detected and position stored
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add free cells
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(
            CellDescription().id(100 + i).pos({100.0f + (i % 3) * 2.0f, 50.0f + (i / 3) * 2.0f}).cellType(FreeCellDescription()).energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Verify lastMatch was stored
    auto sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());

    // Second scan - target hasn't moved, should still be found via relocation
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Verify lastMatch is still present
    sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());

    // Note: During relocation, density is set to 0 (not recalculated)
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorDensity]));
}

TEST_F(SensorTests, detectFreeCell_relocation_targetMoved)
{
    // First scan - target is detected and position stored
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add free cells
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(
            CellDescription().id(100 + i).pos({100.0f + (i % 3) * 2.0f, 50.0f + (i / 3) * 2.0f}).cellType(FreeCellDescription()).energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Get the initial distance
    auto initialDistance = actualSensor._signal->_channels[Channels::SensorDistance];

    // Move the target free cells
    actualData = _simulationFacade->getSimulationData();
    for (int i = 0; i < 10; ++i) {
        actualData.getCellRef(100 + i)._pos.x += 10.0f;  // Move right by 10 units
    }
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target has moved, should be relocated via tracking
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Distance should be different since target moved
    auto newDistance = actualSensor._signal->_channels[Channels::SensorDistance];
    EXPECT_FALSE(approxCompare(initialDistance, newDistance));

    // Verify lastMatch position was updated
    auto sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());
}

TEST_F(SensorTests, detectFreeCell_relocation_targetDisappeared)
{
    // First scan - target is detected and position stored
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add free cells
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(
            CellDescription().id(100 + i).pos({100.0f + (i % 3) * 2.0f, 50.0f + (i / 3) * 2.0f}).cellType(FreeCellDescription()).energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Remove the free cells
    actualData = _simulationFacade->getSimulationData();
    actualData._cells.erase(
        std::remove_if(actualData._cells.begin(), actualData._cells.end(), 
            [](auto const& cell) { 
                return cell._id >= 100 && cell._id < 110 && std::holds_alternative<FreeCellDescription>(cell._cellType); 
            }),
        actualData._cells.end());
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target disappeared, should not be found
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Verify lastMatch was cleared
    auto sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

TEST_F(SensorTests, detectFreeCell_relocation_targetBlocked)
{
    // First scan - target is detected and position stored
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(0.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    // Add free cells
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(
            CellDescription().id(100 + i).pos({100.0f + (i % 3) * 2.0f, 50.0f + (i / 3) * 2.0f}).cellType(FreeCellDescription()).energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Add structure cells between sensor and target to block the ray
    actualData = _simulationFacade->getSimulationData();
    for (int i = 0; i < 10; ++i) {
        actualData._cells.emplace_back(CellDescription().id(50 + i).pos({95.0f + i, 75.0f}).cellType(StructureCellDescription()));
    }
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target is now blocked by structure cells
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Verify lastMatch was cleared
    auto sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

/**
 * Tests for SensorMode_DetectCreature (mode-specific tests)
 */

TEST_F(SensorTests, detectCreature_initialScan_found)
{
    auto data = Description().addCreature(CreatureDescription().id(0).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription())),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addConnection(1, 2);
    data.add(createLargeCreature());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorDensity]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 0.0f);

    // Verify lastMatch was stored
    auto sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());
}

TEST_F(SensorTests, detectCreature_initialScan_notFound_noCreature)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription())),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Verify lastMatch was NOT stored
    auto sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

TEST_F(SensorTests, detectCreature_relocation_creatureStationary)
{
    // First scan - creature is detected and position stored
    auto data = Description().addCreature(CreatureDescription().id(0).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription())),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addConnection(1, 2);
    data.add(createLargeCreature());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Second scan - creature hasn't moved, should still be tracked
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Verify lastMatch is still present
    auto sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());
}

TEST_F(SensorTests, detectCreature_relocation_creatureMoved)
{
    // First scan - creature is detected and position stored
    auto data = Description().addCreature(CreatureDescription().id(0).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription())),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addConnection(1, 2);
    data.add(createLargeCreature(), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Get the initial distance
    auto initialDistance = actualSensor._signal->_channels[Channels::SensorDistance];

    // Move the target creature slightly
    actualData = _simulationFacade->getSimulationData();
    for (int i = 0; i < 100; ++i) {
        actualData.getCellRef(10 + i)._pos.x += 10.0f;  // Move right by 10 units
    }
    _simulationFacade->setSimulationData(actualData);

    // Second scan - creature has moved, should be relocated via tracking
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Distance should be different since creature moved
    auto newDistance = actualSensor._signal->_channels[Channels::SensorDistance];
    EXPECT_FALSE(approxCompare(initialDistance, newDistance));

    // Verify lastMatch position was updated
    auto sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());
}

TEST_F(SensorTests, detectCreature_relocation_creatureDisappeared)
{
    // First scan - creature is detected and position stored
    auto data = Description().addCreature(CreatureDescription().id(0).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription())),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addConnection(1, 2);
    data.add(createLargeCreature(), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Remove the target creature
    actualData._creatures.erase(
        std::remove_if(actualData._creatures.begin(), actualData._creatures.end(), [](auto const& creature) { return creature._id == 1; }),
        actualData._creatures.end());
    _simulationFacade->setSimulationData(actualData);

    // Second scan - creature disappeared, should not be found
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Verify lastMatch was cleared
    auto sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

TEST_F(SensorTests, detectCreature_relocation_creatureBlocked)
{
    // First scan - creature is detected and position stored
    auto data = Description().addCreature(CreatureDescription().id(0).cells({
        CellDescription().id(1).pos({100.0f, 110.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription())),
        CellDescription().id(2).pos({101.0f, 110.0f}),
    }));
    data.addConnection(1, 2);
    data.add(createLargeCreature(), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Add structure cells between sensor and target to block the ray
    actualData = _simulationFacade->getSimulationData();
    for (int i = 0; i < 30; ++i) {
        actualData._cells.emplace_back(CellDescription().id(200 + i).pos({85.0f + i, 100.0f}).cellType(StructureCellDescription()));
    }
    _simulationFacade->setSimulationData(actualData);

    // Second scan - creature is now blocked by structure cells
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));

    // Verify lastMatch was cleared
    auto sensorDesc = std::get<SensorDescription>(actualSensor._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

TEST_F(SensorTests, detectCreature_restrictToColor_found)
{
    auto data = Description().addCreature(CreatureDescription().id(0).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).color(0).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().restrictToColor(1))),
        CellDescription().id(2).pos({101.0f, 100.0f}).color(0),
    }));
    data.addConnection(1, 2);
    
    // Create a large creature with color 1
    std::vector<CellDescription> targetCells;
    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < 10; ++i) {
            targetCells.emplace_back(CellDescription().id(10 + i + j * 10).pos({95.0f + i, 80.0f + j}).color(1));
        }
    }
    data.addCreature(CreatureDescription().id(1).cells(targetCells));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_restrictToColor_notFound)
{
    auto data = Description()
                    .addCreature(CreatureDescription().id(0).cells({
                        CellDescription()
                            .id(1)
                            .pos({100.0f, 100.0f})
                            .frontAngle(0.0f)
                            .color(0)
                            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().restrictToColor(1))),
                        CellDescription().id(2).pos({101.0f, 100.0f}).color(0),
                    }))
                    .addCreature(CreatureDescription().id(1).cells({
                        CellDescription().id(10).pos({100.0f, 50.0f}).color(0),  // Wrong color
                        CellDescription().id(11).pos({101.0f, 50.0f}).color(0),
                    }));
    data.addConnection(1, 2);
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_minNumCells_found)
{
    auto data = Description().addCreature(CreatureDescription().id(0).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().minNumCells(2))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addConnection(1, 2);
    data.add(createLargeCreature());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_minNumCells_notFound)
{
    auto data = Description()
                    .addCreature(CreatureDescription().id(0).cells({
                        CellDescription()
                            .id(1)
                            .pos({100.0f, 100.0f})
                            .frontAngle(0.0f)
                            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().minNumCells(5))),
                        CellDescription().id(2).pos({101.0f, 100.0f}),
                    }))
                    .addCreature(CreatureDescription().id(1).numCells(3).cells({
                        CellDescription().id(10).pos({100.0f, 50.0f}),
                        CellDescription().id(11).pos({101.0f, 50.0f}),
                        CellDescription().id(12).pos({102.0f, 50.0f}),
                    }));
    data.addConnection(1, 2);
    data.addConnection(10, 11);
    data.addConnection(11, 12);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_maxNumCells_found)
{
    auto data = Description().addCreature(CreatureDescription().id(0).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().maxNumCells(200))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addConnection(1, 2);
    data.add(createLargeCreature());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_maxNumCells_notFound)
{
    auto data = Description()
                    .addCreature(CreatureDescription().id(0).cells({
                        CellDescription()
                            .id(1)
                            .pos({100.0f, 100.0f})
                            .frontAngle(0.0f)
                            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().maxNumCells(2))),
                        CellDescription().id(2).pos({101.0f, 100.0f}),
                    }))
                    .addCreature(CreatureDescription().id(1).numCells(3).cells({
                        CellDescription().id(10).pos({100.0f, 50.0f}),
                        CellDescription().id(11).pos({101.0f, 50.0f}),
                        CellDescription().id(12).pos({102.0f, 50.0f}),
                    }));
    data.addConnection(1, 2);
    data.addConnection(10, 11);
    data.addConnection(11, 12);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_restrictToLineage_sameLineage_found)
{
    auto data = Description().addCreature(CreatureDescription().id(0).lineageId(42).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().restrictToLineage(DetectCreatureLineageRestriction_SameLineage))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addConnection(1, 2);
    
    // Create a large creature with same lineage
    std::vector<CellDescription> targetCells;
    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < 10; ++i) {
            targetCells.emplace_back(CellDescription().id(10 + i + j * 10).pos({95.0f + i, 80.0f + j}));
        }
    }
    data.addCreature(CreatureDescription().id(1).lineageId(42).cells(targetCells));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_restrictToLineage_sameLineage_notFound)
{
    auto data = Description()
                    .addCreature(CreatureDescription().id(0).lineageId(42).cells({
                        CellDescription()
                            .id(1)
                            .pos({100.0f, 100.0f})
                            .frontAngle(0.0f)
                            .cellType(SensorDescription().autoTriggerInterval(3).mode(
                                DetectCreatureDescription().restrictToLineage(DetectCreatureLineageRestriction_SameLineage))),
                        CellDescription().id(2).pos({101.0f, 100.0f}),
                    }))
                    .addCreature(CreatureDescription().id(1).lineageId(99).cells({
                        // Different lineage
                        CellDescription().id(10).pos({100.0f, 50.0f}),
                        CellDescription().id(11).pos({101.0f, 50.0f}),
                    }));
    data.addConnection(1, 2);
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_restrictToLineage_otherLineage_found)
{
    auto data = Description().addCreature(CreatureDescription().id(0).lineageId(42).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().restrictToLineage(DetectCreatureLineageRestriction_OtherLineage))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addConnection(1, 2);
    
    // Create a large creature with different lineage
    std::vector<CellDescription> targetCells;
    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < 10; ++i) {
            targetCells.emplace_back(CellDescription().id(10 + i + j * 10).pos({95.0f + i, 80.0f + j}));
        }
    }
    data.addCreature(CreatureDescription().id(1).lineageId(99).cells(targetCells));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_restrictToLineage_otherLineage_notFound)
{
    auto data = Description()
                    .addCreature(CreatureDescription().id(0).lineageId(42).cells({
                        CellDescription()
                            .id(1)
                            .pos({100.0f, 100.0f})
                            .frontAngle(0.0f)
                            .cellType(SensorDescription().autoTriggerInterval(3).mode(
                                DetectCreatureDescription().restrictToLineage(DetectCreatureLineageRestriction_OtherLineage))),
                        CellDescription().id(2).pos({101.0f, 100.0f}),
                    }))
                    .addCreature(CreatureDescription().id(1).lineageId(42).cells({
                        // Same lineage
                        CellDescription().id(10).pos({100.0f, 50.0f}),
                        CellDescription().id(11).pos({101.0f, 50.0f}),
                    }));
    data.addConnection(1, 2);
    data.addConnection(10, 11);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_ignoreStructureCells)
{
    auto data = Description().addCreature(CreatureDescription().id(0).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription())),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addConnection(1, 2);

    // Add structure cells (should be ignored)
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(CellDescription().id(100 + i).pos({100.0f + i, 50.0f}).cellType(StructureCellDescription()));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_ignoreFreeCells)
{
    auto data = Description().addCreature(CreatureDescription().id(0).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription())),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    }));
    data.addConnection(1, 2);

    // Add free cells (should be ignored)
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(CellDescription().id(100 + i).pos({100.0f + i, 50.0f}).cellType(FreeCellDescription()));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_ignoreSameCreature)
{
    auto data = Description().addCreature(CreatureDescription().id(0).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription())),
        CellDescription().id(2).pos({101.0f, 100.0f}),
        CellDescription().id(3).pos({100.0f, 50.0f}),  // Same creature
    }));
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    // Should not detect own creature cells
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_rayBlockedByStructureCells)
{
    auto data = Description().addCreature(CreatureDescription().id(0).cells({
        CellDescription().id(1).pos({100.0f, 110.0f}).frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription())),
        CellDescription().id(2).pos({101.0f, 110.0f}),
    }));
    data.addConnection(1, 2);
    data.add(createLargeCreature(), false);

    // Add structure cells blocking the ray
    for (int i = 0; i < 30; ++i) {
        data._cells.emplace_back(CellDescription().id(200 + i).pos({85.0f + i, 100.0f}).cellType(StructureCellDescription()));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(actualSensor._signal.has_value());
    // Structure cells should block detection
    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}
