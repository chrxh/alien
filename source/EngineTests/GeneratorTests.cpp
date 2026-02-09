#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class GeneratorTests : public IntegrationTestFramework
{
public:
    GeneratorTests()
        : IntegrationTestFramework()
    {}

    ~GeneratorTests() = default;
};

//********************
//* Square Signal    *
//********************

TEST_F(GeneratorTests, squareSignal_timestep0_positiveHalfAmplitude)
{
    // At timestep 0, square signal should be +amplitude/2
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().mode(SquareSignalDesc().amplitude(2.0f).period(10)))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    // Expected: +2.0 / 2 = +1.0
    EXPECT_TRUE(approxCompare(1.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}

TEST_F(GeneratorTests, squareSignal_firstHalfPeriod_positiveHalfAmplitude)
{
    // During first half of period (0 to period/2), signal should be +amplitude/2
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().mode(SquareSignalDesc().amplitude(4.0f).period(100)))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(30);  // 30 < 50 (period/2)

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    // Expected: +4.0 / 2 = +2.0
    EXPECT_TRUE(approxCompare(2.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}

TEST_F(GeneratorTests, squareSignal_secondHalfPeriod_negativeHalfAmplitude)
{
    // During second half of period (period/2 to period-1), signal should be -amplitude/2
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().mode(SquareSignalDesc().amplitude(4.0f).period(100)))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(70);  // 70 >= 50 (period/2)

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    // Expected: -4.0 / 2 = -2.0
    EXPECT_TRUE(approxCompare(-2.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}

TEST_F(GeneratorTests, squareSignal_atHalfPeriod_negativeHalfAmplitude)
{
    // At exactly period/2, signal should switch to negative half
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().mode(SquareSignalDesc().amplitude(2.0f).period(20)))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(11);  // Executes timesteps 0-10, so we get output for timestep 10 (period/2)

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    // Expected: -2.0 / 2 = -1.0
    EXPECT_TRUE(approxCompare(-1.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}

TEST_F(GeneratorTests, squareSignal_wrapsAroundPeriod)
{
    // After completing one period, signal should wrap back to positive half
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().mode(SquareSignalDesc().amplitude(2.0f).period(10)))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(11);  // One timestep into second period

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    // Expected: Back to positive half, +2.0 / 2 = +1.0
    EXPECT_TRUE(approxCompare(1.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}

TEST_F(GeneratorTests, squareSignal_oddPeriod_correctTransition)
{
    // Test with odd period to ensure correct half-period calculation
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().mode(SquareSignalDesc().amplitude(2.0f).period(11)))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    
    // First half: timesteps 0-4 (5 timesteps)
    _simulationFacade->calcTimesteps(4);
    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    EXPECT_TRUE(approxCompare(1.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
    
    // Second half: timesteps 5-10 (6 timesteps)
    _simulationFacade->calcTimesteps(2);  // Now at timestep 6
    actualData = _simulationFacade->getSimulationData();
    generator = actualData.getObjectRef(1);
    EXPECT_TRUE(approxCompare(-1.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}

//********************
//* Sawtooth Signal  *
//********************

TEST_F(GeneratorTests, sawtoothSignal_timestep0_zero)
{
    // At timestep 0, sawtooth signal should start at 0
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().mode(SawtoothSignalDesc().amplitude(10.0f).period(100)))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    // Expected: 10.0 * 0 / 100 = 0.0
    EXPECT_TRUE(approxCompare(0.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}

TEST_F(GeneratorTests, sawtoothSignal_midPeriod_halfAmplitude)
{
    // At mid-period, sawtooth should be approximately half amplitude
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().mode(SawtoothSignalDesc().amplitude(10.0f).period(100)))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(51);  // Executes timesteps 0-50, so we get output for timestep 50

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    // Expected: 10.0 * 50 / 100 = 5.0
    EXPECT_TRUE(approxCompare(5.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}

TEST_F(GeneratorTests, sawtoothSignal_nearEndOfPeriod_nearAmplitude)
{
    // Near end of period, sawtooth should approach amplitude
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().mode(SawtoothSignalDesc().amplitude(10.0f).period(100)))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(100);  // Executes timesteps 0-99, so we get output for timestep 99

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    // Expected: 10.0 * 99 / 100 = 9.9
    EXPECT_TRUE(approxCompare(9.9f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}

TEST_F(GeneratorTests, sawtoothSignal_linearIncrease)
{
    // Verify linear increase across multiple timesteps
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().mode(SawtoothSignalDesc().amplitude(20.0f).period(10)))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    
    for (int timestep = 0; timestep < 10; ++timestep) {
        _simulationFacade->calcTimesteps(1);
        auto actualData = _simulationFacade->getSimulationData();
        auto generator = actualData.getObjectRef(1);
        
        float expected = 20.0f * static_cast<float>(timestep) / 10.0f;
        EXPECT_TRUE(approxCompare(expected, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
    }
}

TEST_F(GeneratorTests, sawtoothSignal_wrapsAroundPeriod)
{
    // After completing one period, signal should wrap back to 0
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().mode(SawtoothSignalDesc().amplitude(10.0f).period(10)))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(11);  // One timestep into second period

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    // Expected: Back to start, 10.0 * 0 / 10 = 0.0
    EXPECT_TRUE(approxCompare(0.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}

//********************
//* Additive Mode    *
//********************

TEST_F(GeneratorTests, squareSignal_nonAdditiveMode_replacesSignal)
{
    // With non-additive mode (default), generator should set the signal value directly
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .type(CellDesc()
                          .cellType(GeneratorDesc().mode(SquareSignalDesc().amplitude(2.0f).period(10)).additive(false))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    // Expected: +2.0 / 2 = +1.0 (set directly, not added)
    EXPECT_TRUE(approxCompare(1.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}

TEST_F(GeneratorTests, squareSignal_additiveMode_behaviorSameAsNonAdditive)
{
    // With additive mode, generator should add to existing signal
    // Since there's no other signal source in this simple test, behavior should be similar
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .type(CellDesc()
                          .cellType(GeneratorDesc().mode(SquareSignalDesc().amplitude(2.0f).period(10)).additive(true))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    // Expected: 0.0 (initial) + 1.0 (generator output) = 1.0
    EXPECT_TRUE(approxCompare(1.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}
