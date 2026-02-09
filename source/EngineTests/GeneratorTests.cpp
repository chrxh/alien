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

struct SquareSignalTestParams
{
    int timesteps;        // Number of timesteps to execute
    float expectedOutput; // Expected signal output
    std::string description;
};

class GeneratorTests_SquareSignal
    : public GeneratorTests
    , public testing::WithParamInterface<SquareSignalTestParams>
{};

// Test square signal at key points in the period
// Period = 100, Amplitude = 4.0
// Expected: +2.0 for timesteps [0, 50), -2.0 for timesteps [50, 100)
INSTANTIATE_TEST_SUITE_P(
    GeneratorTests_SquareSignal,
    GeneratorTests_SquareSignal,
    ::testing::Values(
        SquareSignalTestParams{1, 2.0f, "at the beginning"},              // timestep 0
        SquareSignalTestParams{30, 2.0f, "before halfway through"},        // timestep 29
        SquareSignalTestParams{51, -2.0f, "at halfway through"},           // timestep 50
        SquareSignalTestParams{80, -2.0f, "before the end"},               // timestep 79
        SquareSignalTestParams{100, -2.0f, "at the end"},                  // timestep 99
        SquareSignalTestParams{101, 2.0f, "after the end (wrapping)"}));  // timestep 0 (wrapped)

TEST_P(GeneratorTests_SquareSignal, squareSignal_outputAtVariousTimesteps)
{
    auto params = GetParam();
    
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().mode(SquareSignalDesc().amplitude(4.0f).period(100)))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(params.timesteps);

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    EXPECT_TRUE(approxCompare(params.expectedOutput, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)))
        << "Failed " << params.description << " (after " << params.timesteps << " timesteps)";
}

//********************
//* Sawtooth Signal  *
//********************

struct SawtoothSignalTestParams
{
    int timesteps;        // Number of timesteps to execute
    float expectedOutput; // Expected signal output
    std::string description;
};

class GeneratorTests_SawtoothSignal
    : public GeneratorTests
    , public testing::WithParamInterface<SawtoothSignalTestParams>
{};

// Test sawtooth signal at key points in the period
// Period = 100, Amplitude = 10.0
// Expected: linearly increasing from 0.0 to 10.0 over 100 timesteps
INSTANTIATE_TEST_SUITE_P(
    GeneratorTests_SawtoothSignal,
    GeneratorTests_SawtoothSignal,
    ::testing::Values(
        SawtoothSignalTestParams{1, 0.0f, "at the beginning"},            // timestep 0: 10.0 * 0 / 100
        SawtoothSignalTestParams{30, 2.9f, "before halfway through"},     // timestep 29: 10.0 * 29 / 100
        SawtoothSignalTestParams{51, 5.0f, "at halfway through"},         // timestep 50: 10.0 * 50 / 100
        SawtoothSignalTestParams{80, 7.9f, "before the end"},             // timestep 79: 10.0 * 79 / 100
        SawtoothSignalTestParams{100, 9.9f, "at the end"},                // timestep 99: 10.0 * 99 / 100
        SawtoothSignalTestParams{101, 0.0f, "after the end (wrapping)"}));  // timestep 0 (wrapped): 10.0 * 0 / 100

TEST_P(GeneratorTests_SawtoothSignal, sawtoothSignal_outputAtVariousTimesteps)
{
    auto params = GetParam();
    
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().mode(SawtoothSignalDesc().amplitude(10.0f).period(100)))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(params.timesteps);

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    EXPECT_TRUE(approxCompare(params.expectedOutput, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)))
        << "Failed " << params.description << " (after " << params.timesteps << " timesteps)";
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
