#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
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
    int timesteps;         // Number of timesteps to execute
    int timeOffset;        // Time offset for the generator
    float expectedOutput;  // Expected signal output
    std::string description;
};

class GeneratorTests_SquareSignal
    : public GeneratorTests
    , public testing::WithParamInterface<SquareSignalTestParams>
{};

// Test square signal at key points in the period
// Period = 300 (in timesteps), Amplitude = 2.0, ValueOffset = 0.1
// Expected: +2.1 (clamped to 2.0) for timesteps [0, 150), -1.9 for timesteps [150, 300)
INSTANTIATE_TEST_SUITE_P(
    GeneratorTests_SquareSignal,
    GeneratorTests_SquareSignal,
    ::testing::Values(
        SquareSignalTestParams{3, 0, 2.0f, "at the beginning"},                     // timestep 0: 2.0 + 0.1 = 2.1 clamped to 2.0
        SquareSignalTestParams{90, 0, 2.0f, "before halfway through"},              // timestep 87: 2.0 + 0.1 = 2.1 clamped to 2.0
        SquareSignalTestParams{153, 0, -1.9f, "at halfway through"},                // timestep 150: -2.0 + 0.1 = -1.9
        SquareSignalTestParams{240, 0, -1.9f, "before the end"},                    // timestep 237: -2.0 + 0.1 = -1.9
        SquareSignalTestParams{300, 0, -1.9f, "at the end"},                        // timestep 297: -2.0 + 0.1 = -1.9
        SquareSignalTestParams{303, 0, 2.0f, "after the end (wrapping)"},           // timestep 0 (wrapped): 2.0 + 0.1 = 2.1 clamped to 2.0
        SquareSignalTestParams{3, 150, -1.9f, "with timeOffset at second half"}));  // timeOffset 150: effective pos = 150 => -2.0 + 0.1 = -1.9

TEST_P(GeneratorTests_SquareSignal, squareSignal_outputAtVariousTimesteps)
{
    auto params = GetParam();

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().headCell(true).cellType(
                GeneratorDesc().valueOffset(0.1f).timeOffset(params.timeOffset).mode(SquareSignalDesc().amplitude(2.0f).period(300)))),
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
    int timesteps;         // Number of timesteps to execute
    int timeOffset;        // Time offset for the generator
    float expectedOutput;  // Expected signal output
    std::string description;
};

class GeneratorTests_SawtoothSignal
    : public GeneratorTests
    , public testing::WithParamInterface<SawtoothSignalTestParams>
{};

// Test sawtooth signal at key points in the period
// Period = 300 (in timesteps), Amplitude = 2.0, ValueOffset = 0.2
// Expected: linearly increasing from 0.2 to 2.0 (clamped) over 300 timesteps
INSTANTIATE_TEST_SUITE_P(
    GeneratorTests_SawtoothSignal,
    GeneratorTests_SawtoothSignal,
    ::testing::Values(
        SawtoothSignalTestParams{3, 0, 0.2f, "at the beginning"},                 // timestep 0: 2.0 * 0 / 300 + 0.2 = 0.2
        SawtoothSignalTestParams{90, 0, 0.78f, "before halfway through"},         // timestep 87: 2.0 * 87 / 300 + 0.2 = 0.78
        SawtoothSignalTestParams{153, 0, 1.2f, "at halfway through"},             // timestep 150: 2.0 * 150 / 300 + 0.2 = 1.2
        SawtoothSignalTestParams{240, 0, 1.78f, "before the end"},                // timestep 237: 2.0 * 237 / 300 + 0.2 = 1.78
        SawtoothSignalTestParams{300, 0, 2.0f, "at the end"},                     // timestep 297: 2.0 * 297 / 300 + 0.2 = 2.18 clamped to 2.0
        SawtoothSignalTestParams{303, 0, 0.2f, "after the end (wrapping)"},       // timestep 0 (wrapped): 2.0 * 0 / 300 + 0.2 = 0.2
        SawtoothSignalTestParams{3, 150, 1.2f, "with timeOffset at midpoint"}));  // timeOffset 150: effective pos = 150 => 2.0 * 150 / 300 + 0.2 = 1.2

TEST_P(GeneratorTests_SawtoothSignal, sawtoothSignal_outputAtVariousTimesteps)
{
    auto params = GetParam();

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc().headCell(true).cellType(
                GeneratorDesc().valueOffset(0.2f).timeOffset(params.timeOffset).mode(SawtoothSignalDesc().amplitude(2.0f).period(300)))),
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
            ObjectDesc().id(1).type(CellDesc()
                                        .neuralNetwork(NeuralNetDesc().bias(0, 0.6f))  // Base signal that should be overridden
                                        .cellType(GeneratorDesc().valueOffset(0.15f).mode(SquareSignalDesc().amplitude(1.0f).period(30)).additive(false))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);

    // Expected: 1.0 + 0.15 = 1.15 (set directly, not added to the 0.6 bias)
    EXPECT_TRUE(approxCompare(1.15f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}

TEST_F(GeneratorTests, squareSignal_additiveMode_addsToBaseSignal)
{
    // With additive mode, generator should add to the base signal from the neural network
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc()
                                        .neuralNetwork(NeuralNetDesc().bias(0, 0.6f))  // Base signal that generator adds to
                                        .cellType(GeneratorDesc().valueOffset(0.15f).mode(SquareSignalDesc().amplitude(1.0f).period(30)).additive(true))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);

    // Expected: 0.6 (base from bias) + 1.0 (generator output) + 0.15 (valueOffset) = 1.75
    EXPECT_TRUE(approxCompare(1.75f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}

//**************
//* Truncation *
//**************
TEST_F(GeneratorTests, squareSignal_truncation)
{
    // With additive mode, generator should add to the base signal from the neural network
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).type(CellDesc()
                                        .neuralNetwork(NeuralNetDesc().bias(0, 0.6f))  // Base signal that generator adds to
                                        .cellType(GeneratorDesc().valueOffset(0.15f).mode(SquareSignalDesc().amplitude(2.0f).period(30)).additive(true))),
        },
        CreatureDesc().id(0));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto generator = actualData.getObjectRef(1);

    // Expected: 0.6 (base from bias) + 2.0 (generator output) + 0.15 (valueOffset) = 2.75 truncated to 2.0
    EXPECT_TRUE(approxCompare(2.0f, generator.getCellRef()._signal._channels.at(Channels::GeneratorOutput)));
}
