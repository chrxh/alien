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
};

/**
 * Tests for SensorMode_DetectEnergy
 */

TEST_F(SensorTests, detectEnergy_autoTriggered)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(15).mode(DetectEnergyDescription())),
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

TEST_F(SensorTests, detectEnergy_manuallyTriggered_noSignal)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(std::nullopt).mode(DetectEnergyDescription())),
    });
    _simulationFacade->setSimulationData(data);

    for (int i = 0; i < 10; ++i) {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_FALSE(actualSensor._signal.has_value());
    }
}

TEST_F(SensorTests, detectEnergy_manuallyTriggered_withSignal)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(std::nullopt).mode(DetectEnergyDescription())),
        CellDescription().id(2).pos({101.0f, 100.0f}).signalAndState({1, 0, 0, 0, 0, 0, 0, 0}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
}

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
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(5.0f))),
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
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(50.0f))),
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

TEST_F(SensorTests, detectEnergy_particleAbove)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(5.0f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add particles above the sensor - cluster them to reach minDensity
    for (int i = 0; i < 8; ++i) {
        data._particles.emplace_back(ParticleDescription()
            .id(100 + i)
            .pos({98.0f + i, 20.0f})
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.0f);
    // Angle should be roughly -90 degrees (-0.5 normalized)
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < -0.3f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > -0.7f);
}

TEST_F(SensorTests, detectEnergy_particleAbove_differentFrontAngle)
{
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .frontAngle(90.0f)
            .cellType(SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(5.0f))),
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

TEST_F(SensorTests, detectEnergy_particleBelow)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(5.0f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add particles below the sensor
    for (int i = 0; i < 8; ++i) {
        data._particles.emplace_back(ParticleDescription()
            .id(100 + i)
            .pos({98.0f + i, 180.0f})
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.0f);
    // Angle should be roughly +90 degrees (+0.5 normalized)
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > 0.3f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < 0.7f);
}

TEST_F(SensorTests, detectEnergy_closerParticleDetected)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(5.0f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add a close cluster with high energy
    for (int i = 0; i < 8; ++i) {
        data._particles.emplace_back(ParticleDescription()
            .id(100 + i)
            .pos({98.0f + i, 50.0f})
            .energy(10.0f));
    }
    
    // Add a far cluster with even more energy
    for (int i = 0; i < 12; ++i) {
        data._particles.emplace_back(ParticleDescription()
            .id(200 + i)
            .pos({98.0f + i, 200.0f})
            .energy(15.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    // Should detect the closer particles (above the sensor)
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 0.6f); // Closer = higher value
}

TEST_F(SensorTests, detectEnergy_minRange_found)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(5.0f)).minRange(40)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add particles just beyond minRange
    for (int i = 0; i < 8; ++i) {
        data._particles.emplace_back(ParticleDescription()
            .id(100 + i)
            .pos({98.0f + i, 50.0f})
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectEnergy_minRange_notFound)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(5.0f)).minRange(120)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add particles within minRange (too close)
    for (int i = 0; i < 8; ++i) {
        data._particles.emplace_back(ParticleDescription()
            .id(100 + i)
            .pos({98.0f + i, 50.0f})
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectEnergy_maxRange_found)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(5.0f)).maxRange(120)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add particles within maxRange
    for (int i = 0; i < 8; ++i) {
        data._particles.emplace_back(ParticleDescription()
            .id(100 + i)
            .pos({98.0f + i, 50.0f})
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectEnergy_maxRange_notFound)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(5.0f)).maxRange(30)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add particles beyond maxRange (too far)
    for (int i = 0; i < 8; ++i) {
        data._particles.emplace_back(ParticleDescription()
            .id(100 + i)
            .pos({98.0f + i, 50.0f})
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectEnergy_noParticles)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(5.0f))),
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
            SensorDescription().autoTriggerInterval(3).mode(DetectEnergyDescription().minDensity(5.0f))),
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
 * Tests for SensorMode_DetectFreeCell
 */

TEST_F(SensorTests, detectFreeCell_autoTriggered)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(15).mode(DetectFreeCellDescription())),
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

TEST_F(SensorTests, detectFreeCell_freeCellsFound)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(5.0f))),
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
            SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(50.0f))),
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

TEST_F(SensorTests, detectFreeCell_freeCellsAbove)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(5.0f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add free cells above the sensor
    for (int i = 0; i < 8; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({98.0f + i, 20.0f})
            .cellType(FreeCellDescription())
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.0f);
    // Angle should be roughly -90 degrees (-0.5 normalized)
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < -0.3f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > -0.7f);
}

TEST_F(SensorTests, detectFreeCell_freeCellsBelow)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(5.0f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add free cells below the sensor
    for (int i = 0; i < 8; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({98.0f + i, 180.0f})
            .cellType(FreeCellDescription())
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.0f);
    // Angle should be roughly +90 degrees (+0.5 normalized)
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > 0.3f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < 0.7f);
}

TEST_F(SensorTests, detectFreeCell_closerCellsDetected)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(5.0f))),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add a close cluster
    for (int i = 0; i < 8; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({98.0f + i, 50.0f})
            .cellType(FreeCellDescription())
            .energy(10.0f));
    }
    
    // Add a far cluster with more cells
    for (int i = 0; i < 12; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(200 + i)
            .pos({98.0f + i, 200.0f})
            .cellType(FreeCellDescription())
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    // Should detect the closer free cells (above the sensor)
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 0.6f); // Closer = higher value
}

TEST_F(SensorTests, detectFreeCell_restrictToColor)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).color(0).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(5.0f).restrictToColor(1))),
        CellDescription().id(2).pos({101.0f, 100.0f}).color(0),
    });
    data.addConnection(1, 2);
    
    // Add free cells with wrong color (color 0)
    for (int i = 0; i < 10; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({98.0f + i, 50.0f})
            .color(0)
            .cellType(FreeCellDescription())
            .energy(10.0f));
    }
    
    // Add free cells with correct color (color 1) but fewer
    for (int i = 0; i < 8; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(200 + i)
            .pos({98.0f + i, 150.0f})
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

TEST_F(SensorTests, detectFreeCell_minRange_found)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(5.0f)).minRange(40)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add free cells just beyond minRange
    for (int i = 0; i < 8; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({98.0f + i, 50.0f})
            .cellType(FreeCellDescription())
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectFreeCell_minRange_notFound)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(5.0f)).minRange(120)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add free cells within minRange (too close)
    for (int i = 0; i < 8; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({98.0f + i, 50.0f})
            .cellType(FreeCellDescription())
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectFreeCell_maxRange_found)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(5.0f)).maxRange(60)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add free cells within maxRange
    for (int i = 0; i < 8; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({98.0f + i, 50.0f})
            .cellType(FreeCellDescription())
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectFreeCell_maxRange_notFound)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(5.0f)).maxRange(30)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    
    // Add free cells beyond maxRange
    for (int i = 0; i < 8; ++i) {
        data._cells.emplace_back(CellDescription()
            .id(100 + i)
            .pos({98.0f + i, 50.0f})
            .cellType(FreeCellDescription())
            .energy(10.0f));
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, detectFreeCell_ignoreDifferentCellTypes)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).frontAngle(0.0f).cellType(
            SensorDescription().autoTriggerInterval(3).mode(DetectFreeCellDescription().minDensity(5.0f))),
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
