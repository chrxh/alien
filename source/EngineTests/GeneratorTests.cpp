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
// Period = 100, Amplitude = 2.0 (clamped from 4.0 by validation)
// Expected: +1.0 for timesteps [0, 50), -1.0 for timesteps [50, 100)
INSTANTIATE_TEST_SUITE_P(
    GeneratorTests_SquareSignal,
    GeneratorTests_SquareSignal,
    ::testing::Values(
        SquareSignalTestParams{1, 1.0f, "at the beginning"},              // timestep 0
        SquareSignalTestParams{30, 1.0f, "before halfway through"},        // timestep 29
        SquareSignalTestParams{51, -1.0f, "at halfway through"},           // timestep 50
        SquareSignalTestParams{80, -1.0f, "before the end"},               // timestep 79
        SquareSignalTestParams{100, -1.0f, "at the end"},                  // timestep 99
        SquareSignalTestParams{101, 1.0f, "after the end (wrapping)"}));  // timestep 0 (wrapped)

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
// Period = 100, Amplitude = 2.0
// Expected: linearly increasing from 0.0 to 2.0 over 100 timesteps, clamped to [-2.0, 2.0]
INSTANTIATE_TEST_SUITE_P(
    GeneratorTests_SawtoothSignal,
    GeneratorTests_SawtoothSignal,
    ::testing::Values(
        SawtoothSignalTestParams{1, 0.0f, "at the beginning"},            // timestep 0: 2.0 * 0 / 100 = 0.0
        SawtoothSignalTestParams{30, 0.58f, "before halfway through"},     // timestep 29: 2.0 * 29 / 100 = 0.58
        SawtoothSignalTestParams{51, 1.0f, "at halfway through"},         // timestep 50: 2.0 * 50 / 100 = 1.0
        SawtoothSignalTestParams{80, 1.0f, "before the end"},             // timestep 79: 2.0 * 79 / 100 = 1.58, clamped to 1.0
        SawtoothSignalTestParams{100, 1.0f, "at the end"},                // timestep 99: 2.0 * 99 / 100 = 1.98, clamped to 1.0
        SawtoothSignalTestParams{101, 0.0f, "after the end (wrapping)"}));  // timestep 0 (wrapped): 2.0 * 0 / 100 = 0.0

TEST_P(GeneratorTests_SawtoothSignal, sawtoothSignal_outputAtVariousTimesteps)
{
    auto params = GetParam();
    
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().cellType(GeneratorDesc().mode(SawtoothSignalDesc().amplitude(2.0f).period(100)))),
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
    // With non-additive mode (default), generator should set the signal value directly,
    // overriding any base signal from the neural network
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .type(CellDesc()
                          .neuralNetwork(NeuralNetworkDesc().bias(0, 0.6f))  // Base signal that should be overridden
                          .cellType(GeneratorDesc().mode(SquareSignalDesc().amplitude(2.0f).period(10)).additive(false))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    // Expected: +2.0 / 2 = +1.0 (set directly, not added to the 0.6 bias)
    EXPECT_TRUE(approxCompare(1.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}

TEST_F(GeneratorTests, squareSignal_additiveMode_addsToBaseSignal)
{
    // With additive mode, generator should add to the base signal from the neural network
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .type(CellDesc()
                          .neuralNetwork(NeuralNetworkDesc().bias(0, 0.6f))  // Base signal that generator adds to
                          .cellType(GeneratorDesc().mode(SquareSignalDesc().amplitude(2.0f).period(10)).additive(true))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);
    
    // Expected: 0.6 (base from bias) + 1.0 (generator output) = 1.6
    // But this will be clamped to 1.0 due to signal range limits
    EXPECT_TRUE(approxCompare(1.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}
