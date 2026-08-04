#include <chrono>
#include <cmath>

#include <gtest/gtest.h>

#include <EngineInterface/DescEditService.h>
#include <EngineInterface/Descs.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class NeuronTests : public IntegrationTestFramework
{
public:
    NeuronTests()
        : IntegrationTestFramework()
    {}

    ~NeuronTests() = default;

protected:
    float activationTanh(float value) const { return std::tanh(value); }
    float binaryStep(float value) const { return value >= NEAR_ZERO ? 1.0f : 0.0f; }
    float mod(float value) const { return std::fmod(std::fmod(value + 1.0f, 2.0f) + 2.0f, 2.0f) - 1.0f; }

    float applyActivationFunction(ActivationFunction af, float value)
    {
        switch (af) {
        case ActivationFunction_Tanh:
            return activationTanh(value);
            break;
        case ActivationFunction_BinaryStep:
            return binaryStep(value);
            break;
        case ActivationFunction_Identity:
            return value;
            break;
        case ActivationFunction_Abs:
            return std::abs(value);
            break;
        case ActivationFunction_Gaussian:
            return expf(-2 * value * value);
            break;
        case ActivationFunction_Mod:
            return mod(value);
            break;
        default:
            THROW_NOT_IMPLEMENTED();
        }
    }

    std::vector<float> getExampleSignal1() const { return {0.2f, -0.25f, 0.2f, 0.3f, 0.4f, -0.4f, -0.25, 0.5f}; }
    std::vector<float> getExampleSignal2() const { return {-0.5f, 0.3f, 0.4f, -0.4f, 0, 0, 0.4f, 0.35f}; }
    std::vector<float> addSignals(std::vector<float> const& signal1, std::vector<float> const& signal2)
    {
        CHECK(signal1.size() == signal2.size());
        CHECK(signal1.size() == STANDARD_NEURONS_PER_CELL);
        std::vector<float> result;
        for (int i = 0; i < STANDARD_NEURONS_PER_CELL; ++i) {
            result.emplace_back(signal1.at(i) + signal2.at(i));
        }
        return result;
    }
};

TEST_F(NeuronTests, forwardSignalByDefault)
{
    auto signal1 = getExampleSignal1();
    auto signal2 = getExampleSignal2();

    auto data = ContentDesc()
                    .addCreature({
                        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().neuralActivity(NeuralActivityDesc().signals(signal2))),
                        ObjectDesc().id(2).pos({0, 1}).type(CellDesc().neuralActivity(NeuralActivityDesc().signals(signal1))),
                    })
                    .addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(signal1, actualData.getObjectRef(1).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(signal2, actualData.getObjectRef(2).getCellRef()._neuralActivity._signals));
}

TEST_F(NeuronTests, forwardSignalByDefault_preview)
{
    auto signal1 = getExampleSignal1();
    auto signal2 = getExampleSignal2();

    auto data = ContentDesc()
                    .addCreature({
                        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().neuralActivity(NeuralActivityDesc().signals(signal2))),
                        ObjectDesc().id(2).pos({0, 1}).type(CellDesc().neuralActivity(NeuralActivityDesc().signals(signal1))),
                    })
                    .addConnection(1, 2);

    _simulationFacade->setPreviewData(data);
    _simulationFacade->calcTimestepsForPreview(TIMESTEPS_PER_CELL_FUNCTION, true);
    auto actualData = _simulationFacade->getPreviewData();

    EXPECT_TRUE(approxCompare(signal1, actualData.getObjectRef(1).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(signal2, actualData.getObjectRef(2).getCellRef()._neuralActivity._signals));
}

TEST_F(NeuronTests, emptySignalForZeroConnectionWeight)
{
    auto signal = getExampleSignal1();

    auto data = ContentDesc()
                    .addCreature({
                        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().neuralNetwork(NeuralNetDesc().connectionWeight(0, 0.0f))),
                        ObjectDesc().id(2).pos({0, 1}).type(CellDesc().neuralActivity(signal)),
                    })
                    .addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    std::vector<float> emptySignal(STANDARD_NEURONS_PER_CELL, 0);
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(1).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(2).getCellRef()._neuralActivity._signals));
}

TEST_F(NeuronTests, forkSignal)
{
    auto signal = getExampleSignal1();

    auto data = ContentDesc()
                    .addCreature({
                        ObjectDesc().id(1).pos({1, 2}),
                        ObjectDesc().id(2).pos({2, 2}).type(CellDesc().neuralActivity(NeuralActivityDesc().signals(signal))),
                        ObjectDesc().id(3).pos({3, 2}),
                        ObjectDesc().id(4).pos({2, 3}),
                        ObjectDesc().id(5).pos({2, 1}).type(CellDesc().neuralNetwork(NeuralNetDesc().connectionWeights({0, 1, 0, 0, 0, 0}))),
                        ObjectDesc().id(6).pos({2, 0}),
                    })
                    .addConnection(2, 1)
                    .addConnection(2, 3)
                    .addConnection(2, 4)
                    .addConnection(2, 5)
                    .addConnection(5, 6);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    std::vector<float> emptySignal(STANDARD_NEURONS_PER_CELL, 0);
    EXPECT_TRUE(approxCompare(signal, actualData.getObjectRef(1).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(2).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(signal, actualData.getObjectRef(3).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(signal, actualData.getObjectRef(4).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(5).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(6).getCellRef()._neuralActivity._signals));
}

TEST_F(NeuronTests, mergeSignal)
{
    auto signal1 = getExampleSignal1();
    auto signal2 = getExampleSignal2();

    auto data =
        ContentDesc()
            .addCreature({
                ObjectDesc().id(1).pos({1, 2}).type(CellDesc().neuralActivity(NeuralActivityDesc().signals(signal1))),  // Gets input from cell 2
                ObjectDesc().id(2).pos({2, 2}).type(
                    CellDesc().neuralNetwork(NeuralNetDesc().connectionWeights({1, 0, 1, 1, 0, 0}))),  // Gets input from cell 1, 3, 4 and not cell 5
                ObjectDesc().id(3).pos({3, 2}).type(CellDesc().neuralActivity(NeuralActivityDesc().signals(signal2))),  // Gets input from cell 2
                ObjectDesc().id(4).pos({2, 3}).type(CellDesc().neuralActivity(NeuralActivityDesc().signals(signal2))),  // Gets input from cell 2
                ObjectDesc().id(5).pos({2, 1}).type(CellDesc()
                                                        .neuralActivity(NeuralActivityDesc().signals(signal2))
                                                        .neuralNetwork(NeuralNetDesc().connectionWeights({0, 1, 0, 0, 0, 0}))),  // Gets input from cell 6
                ObjectDesc().id(6).pos({2, 0}),                                                                                  // Gets input from cell 5
            })
            .addConnection(2, 1)
            .addConnection(2, 3)
            .addConnection(2, 4)
            .addConnection(2, 5)
            .addConnection(5, 6);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    std::vector<float> emptySignal(STANDARD_NEURONS_PER_CELL, 0);
    auto sumSignal = addSignals(signal1, signal2);
    sumSignal = addSignals(sumSignal, signal2);
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(1).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(sumSignal, actualData.getObjectRef(2).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(3).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(4).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(5).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(signal2, actualData.getObjectRef(6).getCellRef()._neuralActivity._signals));
}

struct ApplyNeuralNetParameter
{
    ActivationFunction activationFunction;
    int channelIndex;
    float inputValue;
};

inline std::vector<ApplyNeuralNetParameter> generateApplyNeuralNetParameters()
{
    std::vector<ApplyNeuralNetParameter> params;

    for (int af = 0; af < ActivationFunction_Count; ++af) {
        for (int c = 0; c < STANDARD_NEURONS_PER_CELL; ++c) {
            for (int i = 0; i <= 20; ++i) {
                float inputValue = -2.0f + i * 0.2f;
                params.push_back({static_cast<ActivationFunction>(af), c, inputValue});
            }
        }
    }

    return params;
}

class NeuronTests_ApplyNeuralNet
    : public NeuronTests
    , public testing::WithParamInterface<ApplyNeuralNetParameter>
{};

INSTANTIATE_TEST_SUITE_P(NeuronTests_ApplyNeuralNet, NeuronTests_ApplyNeuralNet, ::testing::ValuesIn(generateApplyNeuralNetParameters()));

TEST_P(NeuronTests_ApplyNeuralNet, applyNeuralNet)
{
    auto constexpr ApplyNeuralNetWeight = 0.8f;
    auto constexpr ApplyNeuralNetBias = 0.15f;

    auto param = GetParam();

    // Non-trivial weight and bias values for the test
    // Setup neural network with non-trivial weight matrix and bias vector:
    // - Channel 'c' uses the specified activation function with custom weight and bias
    // - All other channels use Identity activation with identity weight (1.0) and zero bias
    NeuralNetDesc nn;
    for (int i = 0; i < NEURAL_NET_OUTPUTS; ++i) {
        nn._activationFunctions[i] = (i == param.channelIndex) ? param.activationFunction : ActivationFunction_Identity;
        nn.weight(i, i, (i == param.channelIndex) ? ApplyNeuralNetWeight : 1.0f);
        nn._biases[i] = (i == param.channelIndex) ? ApplyNeuralNetBias : 0.0f;
    }

    nn._connectionWeights[0] = 1.0f;  // Enable input signal reception from connected cell

    // Setup input signal:
    // - Channel 'c' has the specified input value
    // - All other channels have 0 input
    std::vector<float> inputSignal(STANDARD_NEURONS_PER_CELL, 0.0f);
    inputSignal[param.channelIndex] = param.inputValue;

    auto data = ContentDesc()
                    .addCreature({
                        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().neuralNetwork(nn)),
                        ObjectDesc().id(2).pos({0, 1}).type(CellDesc().neuralActivity(inputSignal)),
                    })
                    .addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    // Calculate expected output:
    // For test channel 'c':
    //   preActivation = weight * inputValue + bias
    //   output = activation(preActivation), clamped to [-2, 2]
    // All other channels: Identity(0) = 0
    std::vector<float> expected(STANDARD_NEURONS_PER_CELL, 0.0f);

    float preActivation = ApplyNeuralNetWeight * param.inputValue + ApplyNeuralNetBias;
    float rawOutput = applyActivationFunction(param.activationFunction, preActivation);
    expected[param.channelIndex] = std::clamp(rawOutput, -2.0f, 2.0f);

    auto& actual = actualData.getObjectRef(1).getCellRef()._neuralActivity._signals;

    constexpr float precision = 0.1f;
    for (int i = 0; i < STANDARD_NEURONS_PER_CELL; ++i) {
        EXPECT_TRUE(approxCompare(expected[i], actual[i], precision))
            << "Mismatch at channel " << i << ": expected=" << expected[i] << ", actual=" << actual[i];
    }
}

// Test that signals are truncated to [-2, 2] after neural network application
TEST_F(NeuronTests, truncateSignal)
{
    NeuralNetDesc nn;
    for (int i = 0; i < NEURAL_NET_OUTPUTS; ++i) {
        nn.weight(i, i, 2.0f);
    }
    nn._connectionWeights[0] = 1.0f;  // Enable signal forwarding from first connection

    // With weight 2.0f on diagonal, outputs are:
    // Channel 0: 1.5 * 2 = 3.0 -> truncated to 2
    // Channel 3: -1.5 * 2 = -3.0 -> truncated to -2
    // Channel 7: 1.7 * 2 = 3.4 -> truncated to 2
    ContentDesc data;
    data.addCreature({
        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().neuralNetwork(nn)),
        ObjectDesc().id(2).pos({0, 1}).type(CellDesc().neuralActivity({1.5f, 0, 0, -1.5f, 0, 0, 0, 1.7f})),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(std::vector<float>{2, 0, 0, -2, 0, 0, 0, 2}, actualData.getObjectRef(1).getCellRef()._neuralActivity._signals));
}

// Without an open gate on the first input the memory outputs keep their values
TEST_F(NeuronTests, memoryNeuronsRetainActivityAndStayLocal)
{
    std::vector<float> memory = {0.5f, -0.3f, 0.25f, 1.0f};

    auto data = ContentDesc()
                    .addCreature({
                        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().neuralActivity(NeuralActivityDesc().memory(memory))),
                        ObjectDesc().id(2).pos({0, 1}),
                    })
                    .addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    std::vector<float> emptySignal(STANDARD_NEURONS_PER_CELL, 0);
    std::vector<float> emptyMemory(MEMORY_NEURONS_PER_CELL, 0);
    EXPECT_TRUE(approxCompare(memory, actualData.getObjectRef(1).getCellRef()._neuralActivity._memory));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(1).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(2).getCellRef()._neuralActivity._signals));
    EXPECT_TRUE(approxCompare(emptyMemory, actualData.getObjectRef(2).getCellRef()._neuralActivity._memory));
}

TEST_F(NeuronTests, writeMemoryNeuronFromSignal)
{
    auto nn = NeuralNetDesc().weight(STANDARD_NEURONS_PER_CELL, 1, 1.0f);  // First memory neuron reads signal channel 1

    auto data = ContentDesc()
                    .addCreature({
                        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().neuralNetwork(nn)),
                        ObjectDesc().id(2).pos({0, 1}).type(CellDesc().neuralActivity({1.0f, 0.5f, 0, 0, 0, 0, 0, 0})),  // Channel 0 opens the memory gate
                    })
                    .addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    std::vector<float> expectedMemory = {0.5f, 0, 0, 0};
    EXPECT_TRUE(approxCompare(expectedMemory, actualData.getObjectRef(1).getCellRef()._neuralActivity._memory));
}

// A closed memory gate on the first input leaves all memory neurons unchanged
TEST_F(NeuronTests, memoryNeuronsKeepValuesForClosedGate)
{
    std::vector<float> memory = {0.5f, -0.3f, 0.25f, 1.0f};

    NeuralNetDesc nn;
    for (int i = 0; i < MEMORY_NEURONS_PER_CELL; ++i) {
        nn.weight(STANDARD_NEURONS_PER_CELL + i, 1, 1.0f);  // Each memory neuron reads signal channel 1
    }

    auto data = ContentDesc()
                    .addCreature({
                        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().neuralNetwork(nn).neuralActivity(NeuralActivityDesc().memory(memory))),
                        ObjectDesc().id(2).pos({0, 1}).type(CellDesc().neuralActivity({TRIGGER_THRESHOLD, 0.4f, 0, 0, 0, 0, 0, 0})),
                    })
                    .addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(memory, actualData.getObjectRef(1).getCellRef()._neuralActivity._memory));
}

// An open memory gate recalculates all memory neurons, but the gate input itself does not contribute
TEST_F(NeuronTests, memoryNeuronsIgnoreGateInputForOpenGate)
{
    NeuralNetDesc nn;
    for (int i = 0; i < MEMORY_NEURONS_PER_CELL; ++i) {
        nn.weight(STANDARD_NEURONS_PER_CELL + i, 0, 1.0f);                              // Reading the gate input has no effect
        nn.weight(STANDARD_NEURONS_PER_CELL + i, 1, toFloat(i));                        // Each memory neuron reads signal channel 1 with its own weight
        nn.weight(STANDARD_NEURONS_PER_CELL + i, STANDARD_NEURONS_PER_CELL + i, 0.0f);  // Drop the default self-connection
    }

    auto data =
        ContentDesc()
            .addCreature({
                ObjectDesc().id(1).pos({0, 0}).type(CellDesc().neuralNetwork(nn).neuralActivity(NeuralActivityDesc().memory({0.5f, -0.3f, 0.25f, 1.0f}))),
                ObjectDesc().id(2).pos({0, 1}).type(CellDesc().neuralActivity({1.0f, 0.4f, 0, 0, 0, 0, 0, 0})),
            })
            .addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    std::vector<float> expectedMemory = {0.0f, 0.4f, 0.8f, 1.2f};
    EXPECT_TRUE(approxCompare(expectedMemory, actualData.getObjectRef(1).getCellRef()._neuralActivity._memory));
}

TEST_F(NeuronTests, telemetryEnergyInput)
{
    // Signal channel 0 reads the energy telemetry input (1 corresponds to the normal cell energy)
    auto nn = NeuralNetDesc().weight(0, 0, 0.0f).weight(0, STANDARD_NEURONS_PER_CELL + MEMORY_NEURONS_PER_CELL + TelemetryInputs::Energy, 1.0f);

    auto data = ContentDesc()
                    .addCreature({
                        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().neuralNetwork(nn).usableEnergy(150.0f)),
                        ObjectDesc().id(2).pos({0, 1}),
                    })
                    .addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(1.5f, actualData.getObjectRef(1).getCellRef()._neuralActivity._signals.at(0), 0.05f));
}

// Performance test: ~100K connected cells (1000x100 rectangle) for 10000 time steps
class NeuronPerformanceTests : public IntegrationTestFramework
{
public:
    NeuronPerformanceTests()
        : IntegrationTestFramework({2000, 200})  // Large world to fit 1000x100 grid
    {}
};

TEST_F(NeuronPerformanceTests, DISABLED_largeGridPerformance)
{
    constexpr int GridWith = 1000;
    constexpr int GridHeight = 100;
    constexpr int NumCells = GridWith * GridHeight;
    constexpr int NumTimesteps = 10000;

    // Create cells in a 1000x100 grid
    std::vector<ObjectDesc> objects;
    objects.reserve(NumCells);

    for (int y = 0; y < GridHeight; ++y) {
        for (int x = 0; x < GridWith; ++x) {
            uint64_t cellId = static_cast<uint64_t>(y * GridWith + x + 1);
            auto cell = ObjectDesc()
                            .id(cellId)
                            .pos({toFloat(x), toFloat(y)})
                            .type(CellDesc().neuralNetwork(NeuralNetDesc().connectionWeight(0, 1.0f).connectionWeight(1, 1.0f)));
            objects.push_back(cell);
        }
    }

    // Create the description with all cells as a single creature
    ContentDesc data;
    data.addCreature(objects);

    // Create cache for efficient lookups
    auto cache = data.createCache();

    // Add horizontal connections (left-right neighbors)
    for (int y = 0; y < GridHeight; ++y) {
        for (int x = 0; x < GridWith - 1; ++x) {
            uint64_t id1 = static_cast<uint64_t>(y * GridWith + x + 1);
            uint64_t id2 = static_cast<uint64_t>(y * GridWith + (x + 1) + 1);
            data.addConnection(id1, id2, cache);
        }
    }

    // Add vertical connections (up-down neighbors)
    for (int y = 0; y < GridHeight - 1; ++y) {
        for (int x = 0; x < GridWith; ++x) {
            uint64_t id1 = static_cast<uint64_t>(y * GridWith + x + 1);
            uint64_t id2 = static_cast<uint64_t>((y + 1) * GridWith + x + 1);
            data.addConnection(id1, id2, cache);
        }
    }

    _simulationFacade->setSimulationData(data);

    // Run for 10000 time steps
    auto start = std::chrono::high_resolution_clock::now();
    _simulationFacade->calcTimesteps(NumTimesteps);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Performance test: " << NumCells << " cells, " << NumTimesteps << " timesteps completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Average: " << (duration.count() / static_cast<double>(NumTimesteps)) << " ms per timestep" << std::endl;

    // Basic sanity check - simulation should complete without errors
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(actualData.getNumObjects(), NumCells);
}
