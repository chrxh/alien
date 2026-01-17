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
    static SensorModeDescription createModeWithDensity(SensorMode const& mode)
    {
        if (mode == SensorMode_DetectEnergy) {
            return DetectEnergyDescription().minDensity(0.05f);
        } else if (mode == SensorMode_DetectStructure) {
            return DetectStructureDescription();
        } else if (mode == SensorMode_DetectFreeCell) {
            return DetectFreeCellDescription().minDensity(0.05f);
        } else if (mode == SensorMode_DetectCreature) {
            return DetectCreatureDescription();
        } else {
            CHECK(false);
        }
    }

    // Helper to create a large creature (10x10 grid) at the specified position
    Description createLargeCreature(RealVector2D const& startPos = RealVector2D{95.0f, 80.0f}, int diameterCount = 10)
    {
        Description result;
        std::vector<ObjectDescription> target;
        for (int j = 0; j < diameterCount; ++j) {
            for (int i = 0; i < diameterCount; ++i) {
                target.emplace_back(ObjectDescription().id(diameterCount + i + j * diameterCount).pos({startPos.x + i, startPos.y + j}));
            }
        }
        result.addCreature(target, CreatureDescription().id(1));
        for (int j = 0; j < diameterCount; ++j) {
            for (int i = 0; i < diameterCount - 1; ++i) {
                result.addConnection(diameterCount + i + diameterCount * j, diameterCount + 1 + i + diameterCount * j);
            }
        }
        return result;
    }

    // Helper to add detection targets
    void addDetectionTargets(Description& data, SensorMode const& mode, RealVector2D const& startPos, int count = 8, bool assignNewIds = true)
    {
        if (mode == SensorMode_DetectEnergy) {
            for (int i = 0; i < count; ++i) {
                data._energies.emplace_back(EnergyDescription().pos({startPos.x + i, startPos.y}).energy(10.0f));
            }
        } else if (mode == SensorMode_DetectFreeCell) {
            for (int i = 0; i < count; ++i) {
                data._objects.emplace_back(ObjectDescription().pos({startPos.x + i, startPos.y}).type(FreeCellDescription()));
            }
        } else if (mode == SensorMode_DetectStructure) {
            for (int i = 0; i < count; ++i) {
                data._objects.emplace_back(ObjectDescription().pos({startPos.x + i, startPos.y}).type(StructureDescription()));
            }
        } else if (mode == SensorMode_DetectCreature) {
            data.add(createLargeCreature(startPos, count), assignNewIds);
        } else {
            CHECK(false);
        }
    }
};

class SensorTests_AllDetectionModes
    : public SensorTests
    , public testing::WithParamInterface<SensorMode>
{};

INSTANTIATE_TEST_SUITE_P(
    SensorModes,
    SensorTests_AllDetectionModes,
    ::testing::Values(SensorMode_DetectEnergy, SensorMode_DetectStructure, SensorMode_DetectFreeCell, SensorMode_DetectCreature));

class SensorTests_AllDetectionModesExceptStructure
    : public SensorTests
    , public testing::WithParamInterface<SensorMode>
{};

INSTANTIATE_TEST_SUITE_P(
    SensorModes,
    SensorTests_AllDetectionModesExceptStructure,
    ::testing::Values(SensorMode_DetectEnergy, SensorMode_DetectFreeCell, SensorMode_DetectCreature));

/**
 * Parameterized tests that work across all sensor modes
 */

TEST_P(SensorTests_AllDetectionModes, autoTriggered_noTarget)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(15).mode(createModeWithDensity(GetParam())))),
        },
        CreatureDescription());
    _simulationFacade->setSimulationData(data);

    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
        EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
    }
    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
        EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
    }
    {
        _simulationFacade->calcTimesteps(14);
        auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
        EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
    }
}

TEST_P(SensorTests_AllDetectionModes, manuallyTriggered_noSignal)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(
                    CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(std::nullopt).mode(createModeWithDensity(GetParam())))),
        },
        CreatureDescription());
    _simulationFacade->setSimulationData(data);

    for (int i = 0; i < 10; ++i) {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
        EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
    }
}

TEST_P(SensorTests_AllDetectionModes, manuallyTriggered_withSignal)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(
                    CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(std::nullopt).mode(createModeWithDensity(GetParam())))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
        },
        CreatureDescription());
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllDetectionModes, noFrontAngle)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().cellType(SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add detection targets above the sensor
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

TEST_P(SensorTests_AllDetectionModes, targetAbove)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add detection targets above the sensor
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorMass] > 0.0f);
    // Angle should be roughly -90 degrees (-0.5 normalized)
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] < -0.3f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] > -0.7f);
}

TEST_P(SensorTests_AllDetectionModes, targetAbove_differentFrontAngle)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(90.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add detection targets above the sensor
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorMass] > 0.0f);
    // Angle should be roughly -180 degrees
    auto angleAsSignal = actualSensor.getCellRef()._signal._channels[Channels::SensorAngle];
    EXPECT_TRUE(angleAsSignal < -0.9f || angleAsSignal > 0.9f);
}

TEST_P(SensorTests_AllDetectionModes, targetBelow)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add detection targets below the sensor
    addDetectionTargets(data, GetParam(), {98.0f, 180.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorMass] > 0.0f);
    // Angle should be roughly +90 degrees (+0.5 normalized)
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] > 0.3f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] < 0.7f);
}

TEST_P(SensorTests_AllDetectionModes, closerTargetDetected)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add a close cluster above
    addDetectionTargets(data, GetParam(), {100.0f, 50.0f}, 8);

    // Add a far cluster with more items below
    addDetectionTargets(data, GetParam(), {100.0f, 170.0f}, 12);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    // Should detect the closer targets (above the sensor)
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorDistance] > 0.6f);  // Closer = higher value
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorMass] > 0.0f);
    // Angle should be roughly -90 degrees (-0.5 normalized)
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] < -0.3f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] > -0.7f);
}

TEST_P(SensorTests_AllDetectionModes, minRange_found)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(
                    SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())).minRange(40))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add targets just beyond minRange
    addDetectionTargets(data, GetParam(), {98.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllDetectionModes, minRange_notFound)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(
                    SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())).minRange(120))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add targets within minRange (too close)
    addDetectionTargets(data, GetParam(), {98.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    ASSERT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

TEST_P(SensorTests_AllDetectionModes, maxRange_found)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(
                    SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())).maxRange(120))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add targets within maxRange
    addDetectionTargets(data, GetParam(), {98.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllDetectionModes, maxRange_notFound)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(
                    SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())).maxRange(30))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add targets beyond maxRange (too far)
    addDetectionTargets(data, GetParam(), {98.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    ASSERT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

TEST_P(SensorTests_AllDetectionModes, rayBlockedBySameCreatureConnections)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())))),

            // Create a connection that crosses the ray path
            ObjectDescription().id(2).pos({99.0f, 99.0f}),
            ObjectDescription().id(3).pos({101.0f, 99.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(1, 3);

    // Add detection targets above the sensor
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

TEST_P(SensorTests_AllDetectionModes, rayNotBlockedByDifferentCreature)
{
    auto data =
        Description()
            .addCreature(
                {ObjectDescription()
                     .id(1)
                     .pos({100.0f, 100.0f})
                     .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())))),
                 ObjectDescription().id(2).pos({101.0f, 100.0f})},
                CreatureDescription().id(0))
            .addCreature(
                {
                    // Create a different creature with a connection that would cross the ray path
                    ObjectDescription().id(10).pos({100.0f, 99.0f}),
                    ObjectDescription().id(11).pos({101.0f, 99.0f}),
                },
                CreatureDescription());
    data.addConnection(1, 2);    // Sensor creature
    data.addConnection(10, 11);  // Different creature

    // Add detection targets above the sensor
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    // Ray should NOT be blocked by different creature connections
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, rayBlockedByStructureObjects)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add structure cells between sensor and target (to block the ray)
    for (int i = 0; i < 10; ++i) {
        data._objects.emplace_back(ObjectDescription().id(50 + i).pos({95.0f + i, 50.0f}).type(StructureDescription()));
    }

    // Add target behind the structure cells
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);

    // Should not find target because ray is blocked by structure cells
    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, rayNotBlockedByStructureObjects_behind)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add structure cells behind sensor and target
    for (int i = 0; i < 10; ++i) {
        data._objects.emplace_back(ObjectDescription().id(50 + i).pos({95.0f + i, 5.0f}).type(StructureDescription()));
    }

    // Add target behind the structure cells
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorMass] > 0.0f);
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, rayNotBlockedByStructureObjects_differentAngle)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add structure cells behind sensor and target
    for (int i = 0; i < 10; ++i) {
        data._objects.emplace_back(ObjectDescription().id(50 + i).pos({95.0f + i, 150.0f}).type(StructureDescription()));
    }

    // Add target at different angle (not blocking)
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorMass] > 0.0f);
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, relocation_targetStationary)
{
    // First scan - target is detected and position stored
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    addDetectionTargets(data, GetParam(), {100.0f, 60.0f});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    CHECK(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Verify lastMatch was stored
    auto sensorDesc = std::get<SensorDescription>(actualSensor.getCellRef()._cellType);
    CHECK(sensorDesc._lastMatch.has_value());

    // Second scan - target hasn't moved, should still be found via relocation
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Angle should be roughly -90 degrees (-0.5 normalized)
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] < -0.3f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] > -0.7f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorMass] > 0.0f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorDistance] < 0.9f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorDistance] > 0.7f);

    // Verify lastMatch is still present
    sensorDesc = std::get<SensorDescription>(actualSensor.getCellRef()._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, relocation_targetMoved)
{
    // First scan - target is detected and position stored
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add target
    addDetectionTargets(data, GetParam(), {100.0f, 50.0f}, 10, false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    CHECK(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Move the target by deleting and re-adding at a new position
    actualData = _simulationFacade->getSimulationData();
    auto sensorCell = actualData.getObjectRef(1);
    auto auxCell = actualData.getObjectRef(2);
    actualData.clear();
    actualData._objects.emplace_back(sensorCell);
    actualData._objects.emplace_back(auxCell);
    addDetectionTargets(actualData, GetParam(), {120.0f, 30.0f}, 10, false);
    addDetectionTargets(actualData, GetParam(), {100.0f, 120.0f}, 10, true);  // Add a closer target which should be ignored
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target has moved, should be relocated via tracking
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Angle should be roughly -65 degrees (~ -0.4 normalized)
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] < -0.3f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] > -0.5f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorDistance] < 0.8f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorDistance] > 0.5f);

    // Verify lastMatch position was updated
    auto sensorDesc = std::get<SensorDescription>(actualSensor.getCellRef()._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, relocation_targetMoved_aboveMaxRange)
{
    // First scan - target is detected and position stored (within maxRange of 60)
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(
                    SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())).maxRange(60))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add target at distance ~50 (within maxRange of 60)
    addDetectionTargets(data, GetParam(), {100.0f, 50.0f}, 10, false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    CHECK(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Verify lastMatch was stored
    auto sensorDesc = std::get<SensorDescription>(actualSensor.getCellRef()._cellType);
    CHECK(sensorDesc._lastMatch.has_value());

    // Move the target to a position outside the sensor's maxRange
    // New position will be at ~(100, 30), distance from sensor ~70 (> maxRange of 60)
    // But still within relocation search area (+-32 from last match at ~(100, 50))
    actualData = _simulationFacade->getSimulationData();
    auto sensorCell = actualData.getObjectRef(1);
    auto auxCell = actualData.getObjectRef(2);
    actualData.clear();
    actualData._objects.emplace_back(sensorCell);
    actualData._objects.emplace_back(auxCell);
    addDetectionTargets(actualData, GetParam(), {100.0f, 30.0f}, 10, false);
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target has moved outside sensor range, should NOT be found
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);

    // Verify lastMatch was cleared
    sensorDesc = std::get<SensorDescription>(actualSensor.getCellRef()._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, relocation_targetMoved_belowMinRange)
{
    // First scan - target is detected and position stored (beyond minRange of 40)
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(
                    SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())).minRange(40))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add target at distance ~50 (beyond minRange of 40)
    addDetectionTargets(data, GetParam(), {100.0f, 50.0f}, 10, false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    CHECK(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Verify lastMatch was stored
    auto sensorDesc = std::get<SensorDescription>(actualSensor.getCellRef()._cellType);
    CHECK(sensorDesc._lastMatch.has_value());

    // Move the target to a position inside the sensor's minRange
    // New position will be at ~(100, 80), distance from sensor ~20 (< minRange of 40)
    // But still within relocation search area (+-32 from last match at ~(100, 50))
    actualData = _simulationFacade->getSimulationData();
    auto sensorCell = actualData.getObjectRef(1);
    auto auxCell = actualData.getObjectRef(2);
    actualData.clear();
    actualData._objects.emplace_back(sensorCell);
    actualData._objects.emplace_back(auxCell);
    addDetectionTargets(actualData, GetParam(), {100.0f, 80.0f}, 10, false);
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target has moved inside minRange, should NOT be found
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);

    // Verify lastMatch was cleared
    sensorDesc = std::get<SensorDescription>(actualSensor.getCellRef()._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, relocation_targetMoved_forceInitialScan)
{
    // First scan - target is detected and position stored
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription()
                          .frontAngle(0.0f)
                          .neuralNetwork(NeuralNetworkDescription().weight(0, 0, -1.0f))
                          .cellType(SensorDescription().autoTriggerInterval(std::nullopt).mode(createModeWithDensity(GetParam())))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().cellType(GeneratorDescription().autoTriggerInterval(3))),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add target
    addDetectionTargets(data, GetParam(), {100.0f, 50.0f}, 10, false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    CHECK(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Move the target by deleting and re-adding at a new position
    actualData = _simulationFacade->getSimulationData();
    auto sensorCell = actualData.getObjectRef(1);
    auto auxCell = actualData.getObjectRef(2);
    actualData.clear();
    actualData._objects.emplace_back(sensorCell);
    actualData._objects.emplace_back(auxCell);
    addDetectionTargets(actualData, GetParam(), {120.0f, 80.0f}, 10, false);
    addDetectionTargets(actualData, GetParam(), {100.0f, 120.0f}, 10, true);  // Add a closer target which should matched by initial scan
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target has moved, should be relocated via tracking
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Angle should be roughly +90 degrees (+0.5 normalized)
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] > 0.3f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] < 0.7f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorDistance] > 0.8f);

    // Verify lastMatch position was updated
    auto sensorDesc = std::get<SensorDescription>(actualSensor.getCellRef()._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, relocation_targetDisappeared)
{
    // First scan - target is detected and position stored
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add target
    addDetectionTargets(data, GetParam(), {100.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    CHECK(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Move the target by deleting and re-adding at a new position
    actualData = _simulationFacade->getSimulationData();
    auto sensorCell = actualData.getObjectRef(1);
    auto auxCell = actualData.getObjectRef(2);
    actualData.clear();
    actualData._objects.emplace_back(sensorCell);
    actualData._objects.emplace_back(auxCell);
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target disappeared, should not be found
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);

    // Verify lastMatch was cleared
    auto sensorDesc = std::get<SensorDescription>(actualSensor.getCellRef()._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, relocation_targetBlocked)
{
    // First scan - target is detected and position stored
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(createModeWithDensity(GetParam())))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add energy particles above the sensor
    addDetectionTargets(data, GetParam(), {100.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Add structure cells between sensor and target to block the ray
    actualData = _simulationFacade->getSimulationData();
    for (int i = 0; i < 30; ++i) {
        actualData._objects.emplace_back(ObjectDescription().id(50 + i).pos({85.0f + i, 70.0f}).type(StructureDescription()));
    }
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target is now blocked by structure cells
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);

    // Verify lastMatch was cleared
    auto sensorDesc = std::get<SensorDescription>(actualSensor.getCellRef()._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

/**
 * Tests for SensorMode_DetectEnergy (mode-specific tests)
 */
TEST_F(SensorTests, detectEnergy_targetNotFound_belowMinDensity)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(5.0f)))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add a particle with low energy
    data._energies.emplace_back(EnergyDescription().id(100).pos({100.0f, 50.0f}).energy(1.0f));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    ASSERT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

/**
 * Tests for SensorMode_DetectStructure (mode-specific tests)
 */
TEST_F(SensorTests, detectStructure_ignoreDifferentCellTypes)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectStructureDescription()))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add many non-structure cells (should be ignored) - these are creature cells, not structure objects
    std::vector<ObjectDescription> nonStructureCells;
    for (int i = 0; i < 20; ++i) {
        nonStructureCells.emplace_back(
            ObjectDescription().id(100 + i).pos({98.0f + (i % 4), 50.0f + (i / 4)}).type(CellDescription().cellType(BaseDescription()).usableEnergy(10.0f)));
    }
    data.addCreature(nonStructureCells, CreatureDescription());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Should not find anything because only non-structure cells are present
    ASSERT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

/**
 * Tests for SensorMode_DetectFreeCell (mode-specific tests)
 */
TEST_F(SensorTests, detectFreeCell_notFound_belowMinDensity)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(
                    CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.5f)))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add just a few free cells
    data._objects.emplace_back(ObjectDescription().id(100).pos({100.0f, 50.0f}).type(FreeCellDescription()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

TEST_F(SensorTests, detectFreeCell_restrictToColor)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .color(0)
                .type(CellDescription().frontAngle(0.0f).cellType(
                    SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f).restrictToColor(1)))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}).color(0),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add free cells with wrong color (color 0) closer
    for (int i = 0; i < 10; ++i) {
        data._objects.emplace_back(ObjectDescription().id(100 + i).pos({98.0f + i, 80.0f}).color(0).type(FreeCellDescription()));
    }

    // Add free cells with correct color (color 1) farther but still in range
    for (int i = 0; i < 8; ++i) {
        data._objects.emplace_back(ObjectDescription().id(200 + i).pos({98.0f + i, 150.0f}).color(1).type(FreeCellDescription()));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    // Should detect the color 1 cells, not the color 0 cells
    // Color 1 cells are farther (below at y=150 vs y=80), so distance should be higher
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorDistance] > 0.3f);
}

TEST_F(SensorTests, detectFreeCell_ignoreDifferentCellTypes)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(
                    SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(0.05f)))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    // Add many non-free cells (should be ignored) - these are creature cells
    std::vector<ObjectDescription> nonFreeCells;
    for (int i = 0; i < 20; ++i) {
        nonFreeCells.emplace_back(
            ObjectDescription().id(100 + i).pos({98.0f + (i % 4), 50.0f + (i / 4)}).type(CellDescription().cellType(BaseDescription()).usableEnergy(10.0f)));
    }
    data.addCreature(nonFreeCells, CreatureDescription());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Should not find anything because only non-free cells are present
    ASSERT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

/**
 * Tests for SensorMode_DetectCreature (mode-specific tests)
 */
TEST_F(SensorTests, detectCreature_restrictToColor_found)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .color(0)
                .type(CellDescription().frontAngle(0.0f).cellType(
                    SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().restrictToColor(1)))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}).color(0),
        },
        CreatureDescription().id(0));
    data.addConnection(1, 2);

    // Create a large creature with color 1
    std::vector<ObjectDescription> targetCells;
    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < 10; ++i) {
            targetCells.emplace_back(ObjectDescription().id(10 + i + j * 10).pos({95.0f + i, 80.0f + j}).color(1));
        }
    }
    data.addCreature(targetCells, CreatureDescription().id(1));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_restrictToColor_notFound)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .color(0)
                .type(CellDescription().frontAngle(0.0f).cellType(
                    SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().restrictToColor(1)))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}).color(0),
        },
        CreatureDescription().id(0));
    data.addConnection(1, 2);

    data.add(createLargeCreature());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

TEST_F(SensorTests, detectCreature_minNumCells_found)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().minNumCells(2)))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription().id(0));
    data.addConnection(1, 2);

    data.add(createLargeCreature());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_minNumCells_notFound)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(
                    CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().minNumCells(105)))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription().id(0));
    data.addConnection(1, 2);

    data.add(createLargeCreature());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

TEST_F(SensorTests, detectCreature_maxNumCells_found)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(
                    CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().maxNumCells(200)))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription().id(0));
    data.addConnection(1, 2);

    data.add(createLargeCreature());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_maxNumCells_notFound)
{
    auto data = Description()
                    .addCreature(
                        {
                            ObjectDescription()
                                .id(1)
                                .pos({100.0f, 100.0f})
                                .type(CellDescription().frontAngle(0.0f).cellType(
                                    SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().maxNumCells(99)))),
                            ObjectDescription().id(2).pos({101.0f, 100.0f}),
                        },
                        CreatureDescription().id(0))
                    .addCreature(
                        {
                            ObjectDescription().id(10).pos({100.0f, 50.0f}),
                            ObjectDescription().id(11).pos({101.0f, 50.0f}),
                            ObjectDescription().id(12).pos({102.0f, 50.0f}),
                        },
                        CreatureDescription().id(1).numObjects(3));
    data.addConnection(1, 2);

    data.add(createLargeCreature());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

TEST_F(SensorTests, detectCreature_restrictToLineage_sameLineage_found)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(
                    SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().restrictToLineage(LineageRestriction_SameLineage)))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription().id(0).lineageId(42));
    data.addConnection(1, 2);

    // Create a large creature with same lineage
    auto creatureData = createLargeCreature();
    creatureData._creatures.front()._lineageId = 42;
    data.add(std::move(creatureData));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_restrictToLineage_sameLineage_notFound)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(
                    SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().restrictToLineage(LineageRestriction_SameLineage)))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription().id(0).lineageId(42));
    data.addConnection(1, 2);

    // Create a large creature with different lineage
    auto creatureData = createLargeCreature();
    creatureData._creatures.front()._lineageId = 41;
    data.add(std::move(creatureData));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

TEST_F(SensorTests, detectCreature_restrictToLineage_otherLineage_found)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(
                    SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().restrictToLineage(LineageRestriction_OtherLineage)))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription().id(0).lineageId(42));
    data.addConnection(1, 2);

    // Create a large creature with different lineage
    auto creatureData = createLargeCreature();
    creatureData._creatures.front()._lineageId = 41;
    data.add(std::move(creatureData));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_restrictToLineage_otherLineage_notFound)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(
                    SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription().restrictToLineage(LineageRestriction_OtherLineage)))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription().id(0).lineageId(42));
    data.addConnection(1, 2);

    // Create a large creature with same lineage
    auto creatureData = createLargeCreature();
    creatureData._creatures.front()._lineageId = 42;
    data.add(std::move(creatureData));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

TEST_F(SensorTests, detectCreature_ignoreStructureObjects)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription()))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription().id(0));
    data.addConnection(1, 2);

    // Add structure cells (should be ignored)
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            data._objects.emplace_back(ObjectDescription().pos({100.0f + toFloat(i), 50.0f + toFloat(j)}).type(StructureDescription()));
        }
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

TEST_F(SensorTests, detectCreature_ignoreFreeCells)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription()))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription().id(0));
    data.addConnection(1, 2);

    // Add free cells (should be ignored)
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            data._objects.emplace_back(ObjectDescription().pos({100.0f + toFloat(i), 50.0f + toFloat(j)}).type(FreeCellDescription()));
        }
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

TEST_F(SensorTests, detectCreature_ignoreSameCreature)
{
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription()))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
            ObjectDescription().id(3).pos({102.0f, 50.0f}),
            ObjectDescription().id(4).pos({103.0f, 50.0f}),
            ObjectDescription().id(5).pos({104.0f, 50.0f}),
            ObjectDescription().id(6).pos({105.0f, 50.0f}),
            ObjectDescription().id(7).pos({106.0f, 50.0f}),
            ObjectDescription().id(8).pos({107.0f, 50.0f}),
            ObjectDescription().id(9).pos({108.0f, 50.0f}),
            ObjectDescription().id(10).pos({110.0f, 50.0f}),
        },
        CreatureDescription().id(0));
    for (int i = 0; i < 9; ++i) {
        data.addConnection(1 + i, 2 + i);
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Should not detect own creature cells
    EXPECT_FALSE(actualSensor.getCellRef()._signalState == SignalState_Active);
}

/**
 * Tests for SensorMode_DetectCreature cell count density output
 * The density channel should reflect the number of cells in the detected creature
 * using a non-linear scale: 0.5 = 30 cells, 0.75 = 60 cells, 1.0 = 120 cells
 * Formula: density = 0.25 * log2(numCells / 30) + 0.5, clamped to [0, 1]
 */

TEST_F(SensorTests, detectCreature_densityOutputReflectsCellCount_30cells)
{
    // Create sensor creature
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription()))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription().id(0));
    data.addConnection(1, 2);

    // Create a target creature with exactly 30 cells (should give density ~0.5)
    std::vector<ObjectDescription> targetCells;
    for (int i = 0; i < 30; ++i) {
        targetCells.emplace_back(ObjectDescription().id(10 + i).pos({95.0f + (i % 6), 80.0f + (i / 6)}));
    }
    data.addCreature(targetCells, CreatureDescription().id(1));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    // 30 cells should give density ~0.5
    auto density = actualSensor.getCellRef()._signal._channels[Channels::SensorMass];
    EXPECT_TRUE(density > 0.45f);
    EXPECT_TRUE(density < 0.55f);
}

TEST_F(SensorTests, detectCreature_densityOutputReflectsCellCount_60cells)
{
    // Create sensor creature
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription()))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription().id(0));
    data.addConnection(1, 2);

    // Create a target creature with exactly 60 cells (should give density ~0.75)
    std::vector<ObjectDescription> targetCells;
    for (int i = 0; i < 60; ++i) {
        targetCells.emplace_back(ObjectDescription().id(10 + i).pos({95.0f + (i % 8), 80.0f + (i / 8)}));
    }
    data.addCreature(targetCells, CreatureDescription().id(1));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    // 60 cells should give density ~0.75
    auto density = actualSensor.getCellRef()._signal._channels[Channels::SensorMass];
    EXPECT_TRUE(density > 0.70f);
    EXPECT_TRUE(density < 0.80f);
}

TEST_F(SensorTests, detectCreature_densityOutputReflectsCellCount_120cells)
{
    // Create sensor creature
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription()))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription().id(0));
    data.addConnection(1, 2);

    // Create a target creature with exactly 120 cells (should give density ~1.0)
    std::vector<ObjectDescription> targetCells;
    for (int i = 0; i < 120; ++i) {
        targetCells.emplace_back(ObjectDescription().id(10 + i).pos({95.0f + (i % 12), 80.0f + (i / 12)}));
    }
    data.addCreature(targetCells, CreatureDescription().id(1));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    // 120 cells should give density ~1.0
    auto density = actualSensor.getCellRef()._signal._channels[Channels::SensorMass];
    EXPECT_TRUE(density > 0.95f);
}

TEST_F(SensorTests, detectCreature_densityOutputReflectsCellCount_relocation)
{
    // First scan - target creature is detected
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDescription().frontAngle(0.0f).cellType(SensorDescription().autoTriggerInterval(3).mode(DetectCreatureDescription()))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDescription().id(0));
    data.addConnection(1, 2);

    // Create a target creature with 60 cells
    std::vector<ObjectDescription> targetCells;
    for (int i = 0; i < 60; ++i) {
        targetCells.emplace_back(ObjectDescription().id(10 + i).pos({95.0f + (i % 8), 80.0f + (i / 8)}));
    }
    data.addCreature(targetCells, CreatureDescription().id(1));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    CHECK(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Verify initial density is correct (~0.75 for 60 cells)
    auto initialDensity = actualSensor.getCellRef()._signal._channels[Channels::SensorMass];
    EXPECT_TRUE(initialDensity > 0.70f);
    EXPECT_TRUE(initialDensity < 0.80f);

    // Verify lastMatch was stored
    auto sensorDesc = std::get<SensorDescription>(actualSensor.getCellRef()._cellType);
    CHECK(sensorDesc._lastMatch.has_value());

    // Second scan - relocation should also report the correct cell count density
    _simulationFacade->calcTimesteps(3);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    // Relocation should still report density ~0.75 for 60 cells
    auto relocationDensity = actualSensor.getCellRef()._signal._channels[Channels::SensorMass];
    EXPECT_TRUE(relocationDensity > 0.70f);
    EXPECT_TRUE(relocationDensity < 0.80f);
}

/**
 * Tests for SensorMode_Telemetry (mode-specific tests)
 */
TEST_F(SensorTests, telemetry_allOutputs)
{
    // Test with a cell that has energy and velocity to verify all telemetry outputs
    auto data = Description().addCreature(
        {
            ObjectDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .vel({0.1f, 0.05f})
                .type(CellDescription().frontAngle(0.0f).usableEnergy(100.0f).cellType(
                    SensorDescription().autoTriggerInterval(1).mode(TelemetryDescription()))),  // Moving with both x and y components
            ObjectDescription().id(2).pos({101.0f, 100.0f}).vel({0.1f, 0.05f}),
        },
        CreatureDescription());
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);

    ASSERT_TRUE(actualSensor.getCellRef()._signalState == SignalState_Active);

    // Verify cell energy channel is populated and within expected range
    // Formula: 1.0 - 1.0 / pow(energy + 1.0, 0.1)
    // For energy=100: 1.0 - 1.0 / pow(101, 0.1) ≈ 0.36
    auto energySignal = actualSensor.getCellRef()._signal._channels[Channels::SensorTelemetryCellEnergy];
    EXPECT_TRUE(energySignal > 0.3f);
    EXPECT_TRUE(energySignal < 0.4f);

    // Verify velocity angle channel is populated
    // Range: [-1.0, 1.0] representing angle relative to front angle
    auto velAngle = actualSensor.getCellRef()._signal._channels[Channels::SensorTelemetryCellVelAngle];
    EXPECT_TRUE(velAngle >= -1.0f);
    EXPECT_TRUE(velAngle <= 1.0f);

    // Verify velocity strength channel is populated and within expected range
    // Formula: min(log(1.0 + vel * 50) / 1.5, 1.0)
    // For vel ≈ 0.112: log(1 + 5.6) / 1.5 ≈ 0.63
    auto velStrength = actualSensor.getCellRef()._signal._channels[Channels::SensorTelemetryCellVelStrength];
    EXPECT_TRUE(velStrength > 0.5f);
    EXPECT_TRUE(velStrength < 0.7f);
}
