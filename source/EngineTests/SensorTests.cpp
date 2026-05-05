#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
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
    static SensorModeDesc createModeWithDensity(SensorMode const& mode)
    {
        if (mode == SensorMode_DetectEnergy) {
            return DetectEnergyDesc().minDensity(0.05f);
        } else if (mode == SensorMode_DetectSolid) {
            return DetectSolidDesc();
        } else if (mode == SensorMode_DetectFreeCell) {
            return DetectFreeCellDesc().minDensity(0.05f);
        } else if (mode == SensorMode_DetectCreature) {
            return DetectCreatureDesc();
        } else {
            CHECK(false);
        }
    }

    // Helper to create a large creature (10x10 grid) at the specified position
    Desc createLargeCreature(RealVector2D const& startPos = RealVector2D{95.0f, 80.0f}, int diameterCount = 10)
    {
        Desc result;
        std::vector<ObjectDesc> target;
        for (int j = 0; j < diameterCount; ++j) {
            for (int i = 0; i < diameterCount; ++i) {
                target.emplace_back(ObjectDesc().id(diameterCount + i + j * diameterCount).pos({startPos.x + i, startPos.y + j}));
            }
        }
        result.addCreature(target, CreatureDesc().id(1));
        for (int j = 0; j < diameterCount; ++j) {
            for (int i = 0; i < diameterCount - 1; ++i) {
                result.addConnection(diameterCount + i + diameterCount * j, diameterCount + 1 + i + diameterCount * j);
            }
        }
        return result;
    }

    int _nextStructureId = 10000;

    // Helper to add detection targets
    void addDetectionTargets(Desc& data, SensorMode const& mode, RealVector2D const& startPos, int count = 8, bool assignNewIds = true)
    {
        if (mode == SensorMode_DetectEnergy) {
            for (int i = 0; i < count; ++i) {
                data._energies.emplace_back(EnergyDesc().pos({startPos.x + i, startPos.y}).energy(10.0f));
            }
        } else if (mode == SensorMode_DetectFreeCell) {
            for (int i = 0; i < count; ++i) {
                data._objects.emplace_back(ObjectDesc().pos({startPos.x + i, startPos.y}).type(FreeCellDesc()));
            }
        } else if (mode == SensorMode_DetectSolid) {
            auto baseId = _nextStructureId;
            _nextStructureId += count;
            for (int i = 0; i < count; ++i) {
                data._objects.emplace_back(ObjectDesc().id(baseId + i).pos({startPos.x + i, startPos.y}).type(SolidDesc()));
            }
            for (int i = 0; i < count - 1; ++i) {
                data.addConnection(baseId + i, baseId + i + 1);
            }
        } else if (mode == SensorMode_DetectCreature) {
            data.add(createLargeCreature(startPos, count), assignNewIds);
        } else {
            CHECK(false);
        }
    }
};

/**
 * Parameterized tests that work across all sensor modes
 */
class SensorTests_AllDetectionModes
    : public SensorTests
    , public testing::WithParamInterface<SensorMode>
{};

INSTANTIATE_TEST_SUITE_P(
    SensorTests_AllDetectionModes,
    SensorTests_AllDetectionModes,
    ::testing::Values(SensorMode_DetectEnergy, SensorMode_DetectSolid, SensorMode_DetectFreeCell, SensorMode_DetectCreature));

TEST_P(SensorTests_AllDetectionModes, autoTriggered_noTarget)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())))),
        },
        CreatureDesc().id(0));
    _simulationFacade->setSimulationData(data);

    {
        _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);
        auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
        EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    }
    {
        _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);
        auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
        EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    }
    {
        _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);
        auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
        EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    }
}

TEST_P(SensorTests_AllDetectionModes, manuallyTriggered_noSignal)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(false).mode(createModeWithDensity(GetParam())))),
        },
        CreatureDesc().id(0));
    _simulationFacade->setSimulationData(data);

    for (int i = 0; i < 10; ++i) {
        _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);
        auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
        EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    }
}

TEST_P(SensorTests_AllDetectionModes, manuallyTriggered_withSignal)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(false).mode(createModeWithDensity(GetParam())))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().signal({1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);
    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    // Sensor found nothing (no target in range) - result should be 0
    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllDetectionModes, noFrontAngle)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add detection targets above the sensor
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);
    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllDetectionModes, targetAbove)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add detection targets above the sensor
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

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
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(90.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add detection targets above the sensor
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

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
    auto data = Desc()
                    .addCreature(
                        {
                            ObjectDesc()
                                .id(1)
                                .pos({100.0f, 100.0f})
                                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())))),
                            ObjectDesc().id(2).pos({101.0f, 100.0f}),
                        },
                        CreatureDesc().id(0))
                    .addConnection(1, 2);

    // Add detection targets below the sensor
    addDetectionTargets(data, GetParam(), {98.0f, 180.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

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
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add a close cluster above
    addDetectionTargets(data, GetParam(), {100.0f, 50.0f}, 8);

    // Add a far cluster with more items below
    addDetectionTargets(data, GetParam(), {100.0f, 170.0f}, 12);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

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
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())).minRange(40))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add targets just beyond minRange
    addDetectionTargets(data, GetParam(), {98.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllDetectionModes, minRange_notFound)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())).minRange(120))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add targets within minRange (too close)
    addDetectionTargets(data, GetParam(), {98.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllDetectionModes, maxRange_found)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())).maxRange(120))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add targets within maxRange
    addDetectionTargets(data, GetParam(), {98.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllDetectionModes, maxRange_notFound)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())).maxRange(30))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add targets beyond maxRange (too far)
    addDetectionTargets(data, GetParam(), {98.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_nearRangeCornerTargetDetected)
{
    auto data = Desc()
                    .addCreature(
                        {
                            ObjectDesc()
                                .id(1)
                                .pos({100.0f, 100.0f})
                                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectCreatureDesc()).maxRange(8))),
                            ObjectDesc().id(2).pos({101.0f, 100.0f}),
                        },
                        CreatureDesc().id(0))
                    .addConnection(1, 2);

    addDetectionTargets(data, SensorMode_DetectCreature, {108.0f, 108.0f}, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllDetectionModes, rayBlockedBySameCreatureConnections)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())))),

            // Create a connection that crosses the ray path
            ObjectDesc().id(2).pos({99.0f, 99.0f}),
            ObjectDesc().id(3).pos({101.0f, 99.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(1, 3);

    // Add detection targets above the sensor
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllDetectionModes, rayNotBlockedByDifferentCreature)
{
    auto data = Desc()
                    .addCreature(
                        {ObjectDesc()
                             .id(1)
                             .pos({100.0f, 100.0f})
                             .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())))),
                         ObjectDesc().id(2).pos({101.0f, 100.0f})},
                        CreatureDesc().id(0))
                    .addCreature({
                        // Create a different creature with a connection that would cross the ray path
                        ObjectDesc().id(10).pos({100.0f, 99.0f}),
                        ObjectDesc().id(11).pos({101.0f, 99.0f}),
                    });
    data.addConnection(1, 2);    // Sensor creature
    data.addConnection(10, 11);  // Different creature

    // Add detection targets above the sensor
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    // Sensor detected something (signal has non-zero values)
    // Ray should NOT be blocked by different creature connections
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

class SensorTests_AllDetectionModesExceptStructure
    : public SensorTests
    , public testing::WithParamInterface<SensorMode>
{};

INSTANTIATE_TEST_SUITE_P(
    SensorTests_AllDetectionModesExceptStructure,
    SensorTests_AllDetectionModesExceptStructure,
    ::testing::Values(SensorMode_DetectEnergy, SensorMode_DetectFreeCell, SensorMode_DetectCreature));


TEST_P(SensorTests_AllDetectionModesExceptStructure, rayBlockedByStructureObjects)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add solid cells between sensor and target (to block the ray)
    for (int i = 0; i < 10; ++i) {
        data._objects.emplace_back(ObjectDesc().id(50 + i).pos({95.0f + i, 50.0f}).type(SolidDesc()));
    }
    for (int i = 0; i < 9; ++i) {
        data.addConnection(50 + i, 50 + i + 1);
    }

    // Add target behind the solid cells
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);

    // Should not find target because ray is blocked by solid cells
    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, nearRangeRayBlockedByStructureObjects)
{
    auto data = Desc()
                    .addCreature(
                        {
                            ObjectDesc()
                                .id(1)
                                .pos({100.0f, 100.0f})
                                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectCreatureDesc()))),
                            ObjectDesc().id(2).pos({101.0f, 100.0f}),
                        },
                        CreatureDesc().id(0))
                    .addConnection(1, 2);

    data._objects.emplace_back(ObjectDesc().id(50).pos({103.0f, 100.0f}).type(SolidDesc()));
    addDetectionTargets(data, SensorMode_DetectCreature, {106.0f, 100.0f}, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, rayNotBlockedByStructureObjects_behind)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add solid cells behind sensor and target
    for (int i = 0; i < 10; ++i) {
        data._objects.emplace_back(ObjectDesc().id(50 + i).pos({95.0f + i, 5.0f}).type(SolidDesc()));
    }
    for (int i = 0; i < 9; ++i) {
        data.addConnection(50 + i, 50 + i + 1);
    }

    // Add target behind the solid cells
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    // Sensor detected something (signal has non-zero values)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorMass] > 0.0f);
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, rayNotBlockedByStructureObjects_differentAngle)
{
    auto data = Desc()
                    .addCreature(
                        {
                            ObjectDesc()
                                .id(1)
                                .pos({100.0f, 100.0f})
                                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())))),
                            ObjectDesc().id(2).pos({101.0f, 100.0f}),
                        },
                        CreatureDesc().id(0))
                    .addConnection(1, 2);

    // Add solid cells behind sensor and target
    for (int i = 0; i < 10; ++i) {
        data._objects.emplace_back(ObjectDesc().id(50 + i).pos({95.0f + i, 150.0f}).type(SolidDesc()));
    }
    for (int i = 0; i < 9; ++i) {
        data.addConnection(50 + i, 50 + i + 1);
    }

    // Add target at different angle (not blocking)
    addDetectionTargets(data, GetParam(), {98.0f, 20.0f}, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    // Sensor detected something (signal has non-zero values)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorMass] > 0.0f);
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, relocation_targetStationary)
{
    // First scan - target is detected and position stored
    // Using negative signal in channel #0 to enable relocation
    auto data = Desc()
                    .addCreature(
                        {
                            ObjectDesc()
                                .id(1)
                                .pos({100.0f, 100.0f})
                                .type(CellDesc()
                                          .frontAngle(0.0f)
                                          .neuralNetwork(NeuralNetDesc().bias(0, -1.0f))
                                          .cellType(SensorDesc().autoTrigger(false).mode(createModeWithDensity(GetParam())))),
                            ObjectDesc().id(2).pos({101.0f, 100.0f}),
                        },
                        CreatureDesc().id(0))
                    .addConnection(1, 2);

    addDetectionTargets(data, GetParam(), {100.0f, 60.0f});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    CHECK(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Verify lastMatch was stored
    auto sensorDesc = std::get<SensorDesc>(actualSensor.getCellRef()._cellType);
    CHECK(sensorDesc._lastMatch.has_value());

    // Second scan - target hasn't moved, should still be found via relocation
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    // Sensor detected something (signal has non-zero values)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Angle should be roughly -90 degrees (-0.5 normalized)
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] < -0.3f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] > -0.7f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorMass] > 0.0f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorDistance] < 0.9f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorDistance] > 0.7f);

    // Verify lastMatch is still present
    sensorDesc = std::get<SensorDesc>(actualSensor.getCellRef()._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, relocation_targetMoved)
{
    // First scan - target is detected and position stored
    // Using negative signal in channel #0 to enable relocation
    auto data = Desc()
                    .addCreature(
                        {
                            ObjectDesc()
                                .id(1)
                                .pos({100.0f, 100.0f})
                                .type(CellDesc()
                                          .frontAngle(0.0f)
                                          .neuralNetwork(NeuralNetDesc().bias(0, -1.0f))
                                          .cellType(SensorDesc().autoTrigger(false).mode(createModeWithDensity(GetParam())))),
                            ObjectDesc().id(2).pos({101.0f, 100.0f}),
                        },
                        CreatureDesc().id(0))
                    .addConnection(1, 2);

    // Add target
    addDetectionTargets(data, GetParam(), {100.0f, 50.0f}, 10, false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    CHECK(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Move the target by deleting and re-adding at a new position
    actualData = _simulationFacade->getSimulationData();
    auto sensorCell = actualData.getObjectRef(1);
    auto auxCell = actualData.getObjectRef(2);
    auto creature = actualData.getCreatureRef(0);
    auto genome = actualData.getGenomeRef(creature._genomeId);
    actualData.clear();
    actualData._objects.emplace_back(sensorCell);
    actualData._objects.emplace_back(auxCell);
    actualData._creatures.emplace_back(creature);
    actualData._genomes.emplace_back(genome);
    addDetectionTargets(actualData, GetParam(), {120.0f, 30.0f}, 10, false);
    addDetectionTargets(actualData, GetParam(), {100.0f, 120.0f}, 10, true);  // Add a closer target which should be ignored
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target has moved, should be relocated via tracking
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    // Sensor detected something (signal has non-zero values)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Angle should be roughly -65 degrees (~ -0.4 normalized)
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] < -0.3f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] > -0.5f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorDistance] < 0.8f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorDistance] > 0.5f);

    // Verify lastMatch position was updated
    auto sensorDesc = std::get<SensorDesc>(actualSensor.getCellRef()._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, relocation_targetMoved_aboveMaxRange)
{
    // First scan - target is detected and position stored (within maxRange of 60)
    // Using negative signal in channel #0 to enable relocation
    auto data = Desc()
                    .addCreature(
                        {
                            ObjectDesc()
                                .id(1)
                                .pos({100.0f, 100.0f})
                                .type(CellDesc()
                                          .frontAngle(0.0f)
                                          .neuralNetwork(NeuralNetDesc().bias(0, -1.0f))
                                          .cellType(SensorDesc().autoTrigger(false).mode(createModeWithDensity(GetParam())).maxRange(60))),
                            ObjectDesc().id(2).pos({101.0f, 100.0f}),
                        },
                        CreatureDesc().id(0))
                    .addConnection(1, 2);

    // Add target at distance ~50 (within maxRange of 60)
    addDetectionTargets(data, GetParam(), {100.0f, 50.0f}, 10, false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    CHECK(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Verify lastMatch was stored
    auto sensorDesc = std::get<SensorDesc>(actualSensor.getCellRef()._cellType);
    CHECK(sensorDesc._lastMatch.has_value());

    // Move the target to a position outside the sensor's maxRange
    // New position will be at ~(100, 30), distance from sensor ~70 (> maxRange of 60)
    // But still within relocation search area (+-32 from last match at ~(100, 50))
    actualData = _simulationFacade->getSimulationData();
    auto sensorCell = actualData.getObjectRef(1);
    auto auxCell = actualData.getObjectRef(2);
    auto creature = actualData.getCreatureRef(0);
    auto genome = actualData.getGenomeRef(creature._genomeId);
    actualData.clear();
    actualData._objects.emplace_back(sensorCell);
    actualData._objects.emplace_back(auxCell);
    actualData._creatures.emplace_back(creature);
    actualData._genomes.emplace_back(genome);
    addDetectionTargets(actualData, GetParam(), {100.0f, 30.0f}, 10, false);
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target has moved outside sensor range, should NOT be found
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    EXPECT_FALSE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Verify lastMatch was cleared
    sensorDesc = std::get<SensorDesc>(actualSensor.getCellRef()._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, relocation_targetMoved_belowMinRange)
{
    // First scan - target is detected and position stored (beyond minRange of 40)
    // Using negative signal in channel #0 to enable relocation
    auto data = Desc()
                    .addCreature(
                        {
                            ObjectDesc()
                                .id(1)
                                .pos({100.0f, 100.0f})
                                .type(CellDesc()
                                          .frontAngle(0.0f)
                                          .neuralNetwork(NeuralNetDesc().bias(0, -1.0f))
                                          .cellType(SensorDesc().autoTrigger(false).mode(createModeWithDensity(GetParam())).minRange(40))),
                            ObjectDesc().id(2).pos({101.0f, 100.0f}),
                        },
                        CreatureDesc().id(0))
                    .addConnection(1, 2);

    // Add target at distance ~50 (beyond minRange of 40)
    addDetectionTargets(data, GetParam(), {100.0f, 50.0f}, 10, false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    CHECK(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Verify lastMatch was stored
    auto sensorDesc = std::get<SensorDesc>(actualSensor.getCellRef()._cellType);
    CHECK(sensorDesc._lastMatch.has_value());

    // Move the target to a position inside the sensor's minRange
    // New position will be at ~(100, 80), distance from sensor ~20 (< minRange of 40)
    // But still within relocation search area (+-32 from last match at ~(100, 50))
    actualData = _simulationFacade->getSimulationData();
    auto sensorCell = actualData.getObjectRef(1);
    auto auxCell = actualData.getObjectRef(2);
    auto creature = actualData.getCreatureRef(0);
    auto genome = actualData.getGenomeRef(creature._genomeId);
    actualData.clear();
    actualData._objects.emplace_back(sensorCell);
    actualData._objects.emplace_back(auxCell);
    actualData._creatures.emplace_back(creature);
    actualData._genomes.emplace_back(genome);
    addDetectionTargets(actualData, GetParam(), {100.0f, 80.0f}, 10, false);
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target has moved inside minRange, should NOT be found
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    EXPECT_FALSE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Verify lastMatch was cleared
    sensorDesc = std::get<SensorDesc>(actualSensor.getCellRef()._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, relocation_targetMoved_forceInitialScan)
{
    // First scan - target is detected and position stored
    auto data = Desc()
                    .addCreature(
                        {
                            ObjectDesc()
                                .id(1)
                                .pos({100.0f, 100.0f})
                                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(GetParam())))),
                            ObjectDesc().id(2).pos({101.0f, 100.0f}),
                        },
                        CreatureDesc().id(0))
                    .addConnection(1, 2);

    // Add target
    addDetectionTargets(data, GetParam(), {100.0f, 50.0f}, 10, false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    CHECK(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Move the target by deleting and re-adding at a new position
    actualData = _simulationFacade->getSimulationData();
    auto sensorCell = actualData.getObjectRef(1);
    auto auxCell = actualData.getObjectRef(2);
    auto creature = actualData.getCreatureRef(0);
    auto genome = actualData.getGenomeRef(creature._genomeId);
    actualData.clear();
    actualData._objects.emplace_back(sensorCell);
    actualData._objects.emplace_back(auxCell);
    actualData._creatures.emplace_back(creature);
    actualData._genomes.emplace_back(genome);
    addDetectionTargets(actualData, GetParam(), {120.0f, 65.0f}, 10, false);
    addDetectionTargets(actualData, GetParam(), {100.0f, 120.0f}, 10, true);  // Add a closer target which should be matched by initial scan
    _simulationFacade->setSimulationData(actualData);

    // Second scan - relocation disabled via positive signal, should do initial scan and find closer target
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    // Sensor detected something (signal has non-zero values)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Angle should be roughly +90 degrees (+0.5 normalized) - indicating it found the closer target
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] > 0.3f)
        << "SensorAngle: " << actualSensor.getCellRef()._signal._channels[Channels::SensorAngle];
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorAngle] < 0.6f);
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorDistance] > 0.8f);

    // Verify lastMatch position was updated
    auto sensorDesc = std::get<SensorDesc>(actualSensor.getCellRef()._cellType);
    EXPECT_TRUE(sensorDesc._lastMatch.has_value());
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, relocation_targetDisappeared)
{
    // First scan - target is detected and position stored
    // Using negative signal in channel #0 to enable relocation
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .frontAngle(0.0f)
                          .neuralNetwork(NeuralNetDesc().bias(0, -1.0f))
                          .cellType(SensorDesc().autoTrigger(false).mode(createModeWithDensity(GetParam())))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add target
    addDetectionTargets(data, GetParam(), {100.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    CHECK(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Move the target by deleting and re-adding at a new position
    actualData = _simulationFacade->getSimulationData();
    auto sensorCell = actualData.getObjectRef(1);
    auto auxCell = actualData.getObjectRef(2);
    auto creature = actualData.getCreatureRef(0);
    auto genome = actualData.getGenomeRef(creature._genomeId);
    actualData.clear();
    actualData._objects.emplace_back(sensorCell);
    actualData._objects.emplace_back(auxCell);
    actualData._creatures.emplace_back(creature);
    actualData._genomes.emplace_back(genome);
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target disappeared, should not be found
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    EXPECT_FALSE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Verify lastMatch was cleared
    auto sensorDesc = std::get<SensorDesc>(actualSensor.getCellRef()._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

TEST_P(SensorTests_AllDetectionModesExceptStructure, relocation_targetBlocked)
{
    // First scan - target is detected and position stored
    // Using negative signal in channel #0 to enable relocation
    auto data = Desc()
                    .addCreature(
                        {
                            ObjectDesc()
                                .id(1)
                                .pos({100.0f, 100.0f})
                                .type(CellDesc()
                                          .frontAngle(0.0f)
                                          .neuralNetwork(NeuralNetDesc().bias(0, -1.0f))
                                          .cellType(SensorDesc().autoTrigger(false).mode(createModeWithDensity(GetParam())))),
                            ObjectDesc().id(2).pos({101.0f, 100.0f}),
                        },
                        CreatureDesc().id(0))
                    .addConnection(1, 2);

    // Add detection targets above the sensor
    addDetectionTargets(data, GetParam(), {100.0f, 50.0f}, 8);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Add solid cells between sensor and target to block the ray
    actualData = _simulationFacade->getSimulationData();
    for (int i = 0; i < 30; ++i) {
        actualData._objects.emplace_back(ObjectDesc().id(50 + i).pos({85.0f + i, 70.0f}).type(SolidDesc()));
    }
    for (int i = 0; i < 29; ++i) {
        actualData.addConnection(50 + i, 50 + i + 1);
    }
    _simulationFacade->setSimulationData(actualData);

    // Second scan - target is now blocked by solid cells
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    // Sensor did not find a target (blocked)
    EXPECT_FALSE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Verify lastMatch was cleared
    auto sensorDesc = std::get<SensorDesc>(actualSensor.getCellRef()._cellType);
    EXPECT_FALSE(sensorDesc._lastMatch.has_value());
}

/**
 * Tests for SensorMode_DetectEnergy (mode-specific tests)
 */
TEST_F(SensorTests, detectEnergy_targetNotFound_belowMinDensity)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectEnergyDesc().minDensity(5.0f)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add a particle with low energy
    data._energies.emplace_back(EnergyDesc().id(100).pos({100.0f, 50.0f}).energy(1.0f));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

/**
 * Tests for SensorMode_DetectSolid (mode-specific tests)
 */
TEST_F(SensorTests, detectSolid_ignoreDifferentCellTypes)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectSolidDesc()))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add many non-solid cells (should be ignored)
    for (int i = 0; i < 20; ++i) {
        data.addCreature({ObjectDesc().id(100 + i).pos({98.0f + (i % 4), 50.0f + (i / 4)})});
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Should not find anything because only non-solid cells are present
    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectSolid_ignoreFluidParticles)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectSolidDesc()))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add fluid particles - should be ignored
    for (int i = 0; i < 20; ++i) {
        data._objects.emplace_back(ObjectDesc().id(100 + i).pos({98.0f + (i % 4), 50.0f + (i / 4)}).type(FluidDesc()));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Should not find anything because only fluid particles are present
    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, rayNotBlockedByFluidParticles)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectFreeCellDesc().minDensity(0.05f)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add fluid particles between sensor and target - should not block ray
    for (int i = 0; i < 10; ++i) {
        data._objects.emplace_back(ObjectDesc().id(50 + i).pos({95.0f + i, 50.0f}).type(FluidDesc()));
    }

    // Add free cell target behind the fluid particles (large cluster for reliable detection)
    for (int i = 0; i < 12; ++i) {
        data._objects.emplace_back(ObjectDesc().id(200 + i).pos({90.0f + i, 30.0f}).type(FreeCellDesc()));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);

    // Should find target because fluid particles do not block rays
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

/**
 * Tests for SensorMode_DetectFreeCell (mode-specific tests)
 */
TEST_F(SensorTests, detectFreeCell_notFound_belowMinDensity)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectFreeCellDesc().minDensity(0.5f)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add just a few free cells
    data._objects.emplace_back(ObjectDesc().id(100).pos({100.0f, 50.0f}).type(FreeCellDesc()));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);
    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectFreeCell_restrictToColors)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .color(0)
                .type(
                    CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectFreeCellDesc().minDensity(0.05f).restrictToColors(1 << 1)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).color(0),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add free cells with wrong color (color 0) closer
    for (int i = 0; i < 10; ++i) {
        data.addObjects({ObjectDesc().id(100 + i).pos({98.0f + i, 80.0f}).color(0).type(FreeCellDesc())});
    }

    // Add free cells with correct color (color 1) farther but still in range
    for (int i = 0; i < 8; ++i) {
        data.addObjects({ObjectDesc().id(200 + i).pos({98.0f + i, 150.0f}).color(1).type(FreeCellDesc())});
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    // Should detect the color 1 cells, not the color 0 cells
    // Color 1 cells are farther (below at y=150 vs y=80), so distance should be higher
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorDistance] > 0.3f);
}

TEST_F(SensorTests, detectFreeCell_ignoreDifferentCellTypes)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectFreeCellDesc().minDensity(0.05f)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add many non-free cells (should be ignored)
    for (int i = 0; i < 20; ++i) {
        data.addCreature({ObjectDesc().id(100 + i).pos({98.0f + (i % 4), 50.0f + (i / 4)}).type(CellDesc().cellType(BaseDesc()).usableEnergy(10.0f))});
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Should not find anything because only non-free cells are present
    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

/**
 * Tests for SensorMode_DetectCreature (mode-specific tests)
 */
namespace
{
    std::vector<float> getAllAngles()
    {
        std::vector<float> result;
        for (float angle = 0; angle < 360.0f; angle += 5.0f) {
            result.emplace_back(angle);
        }
        return result;
    }
}

class SensorTests_AllAngles
    : public SensorTests
    , public testing::WithParamInterface<float>
{};

INSTANTIATE_TEST_SUITE_P(SensorTests_AllAngles, SensorTests_AllAngles, ::testing::ValuesIn(getAllAngles()));

TEST_P(SensorTests_AllAngles, detectCreature_nearRangeScan)
{
    auto angle = GetParam();

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(createModeWithDensity(SensorMode_DetectCreature)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add detection targets near around the sensor
    addDetectionTargets(data, SensorMode_DetectCreature, RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(angle) * 6.0f, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor.getCellRef()._signal._channels[Channels::SensorMass] > 0.0f);

    // Verify the detected angle matches the expected direction (with tolerance for scan discretization)
    // The absolute angle from sensor to target is 'angle' (input parameter)
    // Reference direction is from sensor (100, 100) to connected cell (101, 100), which is (1, 0) = angle 90
    // Relative angle = absAngle - refAngle - frontAngle = angle - 90 - 0 = angle - 90
    // Signal channels use normalized angle: angle / 180 to get [-1, 1]
    auto expectedRelAngle = Math::getNormalizedAngle(angle - 90.0f, -180.0f);
    auto actualNormalizedAngle = actualSensor.getCellRef()._signal._channels[Channels::SensorAngle];
    auto actualRelAngle = actualNormalizedAngle * 180.0f;
    // Use angle comparison with 20 degree tolerance due to grid-based scan discretization
    EXPECT_TRUE(approxCompareAngles(expectedRelAngle, actualRelAngle, 20.0f))
        << "Expected angle " << expectedRelAngle << " but got " << actualRelAngle << " for input angle " << angle;

    // Verify the detected distance is approximately 6.0 units
    // Distance formula: 1.0 - min(1.0, distance / 256)
    // For distance = 6.0: 1.0 - 6.0/256 ≈ 0.977
    auto expectedDistance = 1.0f - 6.0f / 256.0f;
    auto actualDistance = actualSensor.getCellRef()._signal._channels[Channels::SensorDistance];
    EXPECT_TRUE(approxCompare(expectedDistance, actualDistance, 0.1f)) << "Expected distance " << expectedDistance << " but got " << actualDistance;
}


TEST_F(SensorTests, detectCreature_restrictToColors_found)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .color(0)
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectCreatureDesc().restrictToColors(1 << 1)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).color(0),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Create a large creature with color 1
    std::vector<ObjectDesc> targetCells;
    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < 10; ++i) {
            targetCells.emplace_back(ObjectDesc().id(10 + i + j * 10).pos({95.0f + i, 80.0f + j}).color(1));
        }
    }
    data.addCreature(targetCells, CreatureDesc().id(1));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Sensor detected something (signal has non-zero values)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_restrictToColors_notFound)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .color(0)
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectCreatureDesc().restrictToColors(1 << 1)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).color(0),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    data.add(createLargeCreature());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_minNumCells_found)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectCreatureDesc().minNumCells(2)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    data.add(createLargeCreature());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Sensor detected something (signal has non-zero values)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_minNumCells_notFound)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectCreatureDesc().minNumCells(105)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    data.add(createLargeCreature());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_maxNumCells_found)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectCreatureDesc().maxNumCells(200)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    data.add(createLargeCreature());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Sensor detected something (signal has non-zero values)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_maxNumCells_notFound)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectCreatureDesc().maxNumCells(99)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    data.add(createLargeCreature());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_restrictToLineage_relatedLineage_found)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(
                    SensorDesc().autoTrigger(true).mode(DetectCreatureDesc().restrictToLineage(LineageRestriction_RelatedLineage)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc().lineageId(42));
    data.addConnection(1, 2);

    // Create a large creature with same lineage
    auto creatureData = createLargeCreature();
    creatureData._genomes.front()._lineageId = 42;
    data.add(std::move(creatureData));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Sensor detected something (signal has non-zero values)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_restrictToLineage_relatedLineage2_found)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(
                    SensorDesc().autoTrigger(true).mode(DetectCreatureDesc().restrictToLineage(LineageRestriction_RelatedLineage)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc().lineageId(41).prevLineageId(42));
    data.addConnection(1, 2);

    // Create a large creature with same lineage
    auto creatureData = createLargeCreature();
    creatureData._genomes.front().lineageId(43).prevLineageId(42);
    data.add(std::move(creatureData));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Sensor detected something (signal has non-zero values)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}


TEST_F(SensorTests, detectCreature_restrictToLineage_relatedLineage_notFound)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(
                    SensorDesc().autoTrigger(true).mode(DetectCreatureDesc().restrictToLineage(LineageRestriction_RelatedLineage)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc().lineageId(42));
    data.addConnection(1, 2);

    // Create a large creature with different lineage
    auto creatureData = createLargeCreature();
    creatureData._genomes.front()._lineageId = 41;
    data.add(std::move(creatureData));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_restrictToLineage_unrelatedLineage_found)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(
                    SensorDesc().autoTrigger(true).mode(DetectCreatureDesc().restrictToLineage(LineageRestriction_UnrelatedLineage)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc().lineageId(42));
    data.addConnection(1, 2);

    // Create a large creature with different lineage
    auto creatureData = createLargeCreature();
    creatureData._genomes.front()._lineageId = 41;
    data.add(std::move(creatureData));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Sensor detected something (signal has non-zero values)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_restrictToLineage_unrelatedLineage_notFound)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(
                    SensorDesc().autoTrigger(true).mode(DetectCreatureDesc().restrictToLineage(LineageRestriction_UnrelatedLineage)))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc().lineageId(42));
    data.addConnection(1, 2);

    // Create a large creature with same lineage
    auto creatureData = createLargeCreature();
    creatureData._genomes.front()._lineageId = 42;
    data.add(std::move(creatureData));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_ignoreStructureObjects)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectCreatureDesc()))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add solid cells (should be ignored)
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            data._objects.emplace_back(ObjectDesc().pos({100.0f + toFloat(i), 50.0f + toFloat(j)}).type(SolidDesc()));
        }
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_ignoreFreeCells)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectCreatureDesc()))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Add free cells (should be ignored)
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            data._objects.emplace_back(ObjectDesc().pos({100.0f + toFloat(i), 50.0f + toFloat(j)}).type(FreeCellDesc()));
        }
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectCreature_ignoreSameCreature)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectCreatureDesc()))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
            ObjectDesc().id(3).pos({102.0f, 50.0f}),
            ObjectDesc().id(4).pos({103.0f, 50.0f}),
            ObjectDesc().id(5).pos({104.0f, 50.0f}),
            ObjectDesc().id(6).pos({105.0f, 50.0f}),
            ObjectDesc().id(7).pos({106.0f, 50.0f}),
            ObjectDesc().id(8).pos({107.0f, 50.0f}),
            ObjectDesc().id(9).pos({108.0f, 50.0f}),
            ObjectDesc().id(10).pos({110.0f, 50.0f}),
        },
        CreatureDesc().id(0));
    for (int i = 0; i < 9; ++i) {
        data.addConnection(1 + i, 2 + i);
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Should not detect own creature cells
    EXPECT_TRUE(approxCompare(0.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
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
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectCreatureDesc()))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Create a target creature with exactly 30 cells (should give density ~0.5)
    std::vector<ObjectDesc> targetCells;
    for (int i = 0; i < 30; ++i) {
        targetCells.emplace_back(ObjectDesc().id(10 + i).pos({95.0f + (i % 6), 90.0f + (i / 6)}));
    }
    data.addCreature(targetCells, CreatureDesc().id(1));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Sensor detected something (signal has non-zero values)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    // 30 cells should give density ~0.5
    auto density = actualSensor.getCellRef()._signal._channels[Channels::SensorMass];
    EXPECT_TRUE(density > 0.40f);
    EXPECT_TRUE(density < 0.6f);
}

TEST_F(SensorTests, detectCreature_densityOutputReflectsCellCount_60cells)
{
    // Create sensor creature
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectCreatureDesc()))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Create a target creature with exactly 60 cells (should give density ~0.75)
    std::vector<ObjectDesc> targetCells;
    for (int i = 0; i < 60; ++i) {
        targetCells.emplace_back(ObjectDesc().id(10 + i).pos({95.0f + (i % 8), 80.0f + (i / 8)}));
    }
    data.addCreature(targetCells, CreatureDesc().id(1));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Sensor detected something (signal has non-zero values)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    // 60 cells should give density ~0.75
    auto density = actualSensor.getCellRef()._signal._channels[Channels::SensorMass];
    EXPECT_TRUE(density > 0.70f);
    EXPECT_TRUE(density < 0.80f);
}

TEST_F(SensorTests, detectCreature_densityOutputReflectsCellCount_120cells)
{
    // Create sensor creature
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(DetectCreatureDesc()))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Create a target creature with exactly 120 cells (should give density ~1.0)
    std::vector<ObjectDesc> targetCells;
    for (int i = 0; i < 120; ++i) {
        targetCells.emplace_back(ObjectDesc().id(10 + i).pos({95.0f + (i % 12), 80.0f + (i / 12)}));
    }
    data.addCreature(targetCells, CreatureDesc().id(1));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);

    // Sensor detected something (signal has non-zero values)
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));
    // 120 cells should give density ~1.0
    auto density = actualSensor.getCellRef()._signal._channels[Channels::SensorMass];
    EXPECT_TRUE(density > 0.95f);
}

TEST_F(SensorTests, detectCreature_relocation_densityOutputReflectsCellCount)
{
    // First scan - target creature is detected
    // Using negative signal in channel #0 to enable relocation
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .frontAngle(0.0f)
                          .neuralNetwork(NeuralNetDesc().bias(0, -1.0f))
                          .cellType(SensorDesc().autoTrigger(false).mode(DetectCreatureDesc()))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0));
    data.addConnection(1, 2);

    // Create a target creature with 60 cells
    std::vector<ObjectDesc> targetCells;
    for (int i = 0; i < 60; ++i) {
        targetCells.emplace_back(ObjectDesc().id(10 + i).pos({95.0f + (i % 8), 80.0f + (i / 8)}));
    }
    data.addCreature(targetCells, CreatureDesc().id(1));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getObjectRef(1);
    CHECK(approxCompare(1.0f, actualSensor.getCellRef()._signal._channels[Channels::SensorFoundResult]));

    // Verify initial density is correct (~0.75 for 60 cells)
    auto initialDensity = actualSensor.getCellRef()._signal._channels[Channels::SensorMass];
    EXPECT_TRUE(initialDensity > 0.70f);
    EXPECT_TRUE(initialDensity < 0.80f);

    // Verify lastMatch was stored
    auto sensorDesc = std::get<SensorDesc>(actualSensor.getCellRef()._cellType);
    CHECK(sensorDesc._lastMatch.has_value());

    // Second scan - relocation should also report the correct cell count density
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);  // Wait for next trigger
    actualData = _simulationFacade->getSimulationData();
    actualSensor = actualData.getObjectRef(1);

    // Sensor detected something (signal has non-zero values)
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
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .vel({0.1f, 0.05f})
                .type(CellDesc().frontAngle(0.0f).usableEnergy(100.0f).cellType(
                    SensorDesc().autoTrigger(true).mode(TelemetryDesc()))),  // Moving with both x and y components
        },
        CreatureDesc().id(0));
    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);
    auto actualSensor = _simulationFacade->getSimulationData().getObjectRef(1);

    // Sensor detected something (signal has non-zero values)

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
