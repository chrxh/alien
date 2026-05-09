#include <chrono>
#include <cmath>

#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
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
        default:
            THROW_NOT_IMPLEMENTED();
        }
    }

    std::vector<float> getExampleSignal1() const { return {0.2f, -0.25f, 0.2f, 0.3f, 0.4f, -0.4f, -0.25, 0.5f, 0, 0.4f, 0.2f, 0.2f, -0.1f, 0.05f, 0, 0.25f}; }
    std::vector<float> getExampleSignal2() const { return {-0.5f, 0.3f, 0.4f, -0.4f, 0, 0, 0.4f, 0.35f, 0, 0.4f, 0, -0.2f, 0, 0.2f, 0.2f, -0.5f}; }
    std::vector<float> addSignals(std::vector<float> const& signal1, std::vector<float> const& signal2)
    {
        CHECK(signal1.size() == signal2.size());
        CHECK(signal1.size() == NEURONS_PER_CELL);
        std::vector<float> result;
        for (int i = 0; i < NEURONS_PER_CELL; ++i) {
            result.emplace_back(signal1.at(i) + signal2.at(i));
        }
        return result;
    }
};

TEST_F(NeuronTests, forwardSignalByDefault)
{
    auto signal1 = getExampleSignal1();
    auto signal2 = getExampleSignal2();

    // Set different numTimesSent values on input signals
    auto data = Desc()
                    .addCreature({
                        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().signal(SignalDesc().channels(signal2).numTimesSent(3))),
                        ObjectDesc().id(2).pos({0, 1}).type(CellDesc().signal(SignalDesc().channels(signal1).numTimesSent(7))),
                    })
                    .addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(signal1, actualData.getObjectRef(1).getCellRef()._signal._channels));
    EXPECT_TRUE(approxCompare(signal2, actualData.getObjectRef(2).getCellRef()._signal._channels));

    // Check numTimesSent: cell 1 receives from cell 2 (numTimesSent=7), so its numTimesSent should be 7
    // Cell 2 receives from cell 1 (numTimesSent=3), so its numTimesSent should be 3
    EXPECT_EQ(actualData.getObjectRef(1).getCellRef()._signal._numTimesSent, 7);
    EXPECT_EQ(actualData.getObjectRef(2).getCellRef()._signal._numTimesSent, 3);
}

TEST_F(NeuronTests, forwardSignalByDefault_preview)
{
    auto signal1 = getExampleSignal1();
    auto signal2 = getExampleSignal2();

    // Set different numTimesSent values on input signals
    auto data = Desc()
                    .addCreature({
                        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().signal(SignalDesc().channels(signal2).numTimesSent(2))),
                        ObjectDesc().id(2).pos({0, 1}).type(CellDesc().signal(SignalDesc().channels(signal1).numTimesSent(5))),
                    })
                    .addConnection(1, 2);

    _simulationFacade->setPreviewData(data);
    _simulationFacade->calcTimestepsForPreview(TIMESTEPS_PER_CELL_FUNCTION, true);
    auto actualData = _simulationFacade->getPreviewData();

    EXPECT_TRUE(approxCompare(signal1, actualData.getObjectRef(1).getCellRef()._signal._channels));
    EXPECT_TRUE(approxCompare(signal2, actualData.getObjectRef(2).getCellRef()._signal._channels));

    // Check numTimesSent: cell 1 receives from cell 2 (numTimesSent=5), so its numTimesSent should be 5
    // Cell 2 receives from cell 1 (numTimesSent=2), so its numTimesSent should be 2
    EXPECT_EQ(actualData.getObjectRef(1).getCellRef()._signal._numTimesSent, 5);
    EXPECT_EQ(actualData.getObjectRef(2).getCellRef()._signal._numTimesSent, 2);
}

TEST_F(NeuronTests, emptySignalForZeroConnectionWeight)
{
    auto signal = getExampleSignal1();

    auto data = Desc()
                    .addCreature({
                        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().neuralNetwork(NeuralNetDesc().connectionWeight(0, 0.0f))),
                        ObjectDesc().id(2).pos({0, 1}).type(CellDesc().signal(signal)),
                    })
                    .addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    std::vector<float> emptySignal(NEURONS_PER_CELL, 0);
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(1).getCellRef()._signal._channels));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(2).getCellRef()._signal._channels));
}

TEST_F(NeuronTests, forkSignal)
{
    auto signal = getExampleSignal1();

    // Set numTimesSent on the source signal
    auto data = Desc()
                    .addCreature({
                        ObjectDesc().id(1).pos({1, 2}),
                        ObjectDesc().id(2).pos({2, 2}).type(CellDesc().signal(SignalDesc().channels(signal).numTimesSent(4))),
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

    std::vector<float> emptySignal(NEURONS_PER_CELL, 0);
    EXPECT_TRUE(approxCompare(signal, actualData.getObjectRef(1).getCellRef()._signal._channels));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(2).getCellRef()._signal._channels));
    EXPECT_TRUE(approxCompare(signal, actualData.getObjectRef(3).getCellRef()._signal._channels));
    EXPECT_TRUE(approxCompare(signal, actualData.getObjectRef(4).getCellRef()._signal._channels));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(5).getCellRef()._signal._channels));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(6).getCellRef()._signal._channels));

    // Check numTimesSent: cells 1, 3, 4 receive the signal from cell 2 (numTimesSent=4)
    EXPECT_EQ(actualData.getObjectRef(1).getCellRef()._signal._numTimesSent, 4);
    EXPECT_EQ(actualData.getObjectRef(3).getCellRef()._signal._numTimesSent, 4);
    EXPECT_EQ(actualData.getObjectRef(4).getCellRef()._signal._numTimesSent, 4);
}

TEST_F(NeuronTests, mergeSignal)
{
    auto signal1 = getExampleSignal1();
    auto signal2 = getExampleSignal2();

    // Use different numTimesSent values: signal1 has numTimesSent=3, signal2 has numTimesSent=5, signal2 in cell 4 has numTimesSent=2
    // Cell 2 receives from cells 1 (numTimesSent=3), 3 (numTimesSent=5), 4 (numTimesSent=2)
    // The resulting numTimesSent should be min(3, 5, 2) = 2

    auto data =
        Desc()
            .addCreature({
                ObjectDesc().id(1).pos({1, 2}).type(CellDesc().signal(SignalDesc().channels(signal1).numTimesSent(3))),  // Gets input from cell 2
                ObjectDesc().id(2).pos({2, 2}).type(
                    CellDesc().neuralNetwork(NeuralNetDesc().connectionWeights({1, 0, 1, 1, 0, 0}))),  // Gets input from cell 1, 3, 4 and not cell 5
                ObjectDesc().id(3).pos({3, 2}).type(CellDesc().signal(SignalDesc().channels(signal2).numTimesSent(5))),  // Gets input from cell 2
                ObjectDesc().id(4).pos({2, 3}).type(CellDesc().signal(SignalDesc().channels(signal2).numTimesSent(2))),  // Gets input from cell 2
                ObjectDesc().id(5).pos({2, 1}).type(CellDesc()
                                                        .signal(SignalDesc().channels(signal2).numTimesSent(4))
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

    std::vector<float> emptySignal(NEURONS_PER_CELL, 0);
    auto sumSignal = addSignals(signal1, signal2);
    sumSignal = addSignals(sumSignal, signal2);
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(1).getCellRef()._signal._channels));
    EXPECT_TRUE(approxCompare(sumSignal, actualData.getObjectRef(2).getCellRef()._signal._channels));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(3).getCellRef()._signal._channels));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(4).getCellRef()._signal._channels));
    EXPECT_TRUE(approxCompare(emptySignal, actualData.getObjectRef(5).getCellRef()._signal._channels));
    EXPECT_TRUE(approxCompare(signal2, actualData.getObjectRef(6).getCellRef()._signal._channels));

    // Check numTimesSent on the merged signal
    // Cell 2 receives from cells 1 (numTimesSent=3), 3 (numTimesSent=5), 4 (numTimesSent=2)
    // The resulting numTimesSent should be min(3, 5, 2) = 2
    EXPECT_EQ(actualData.getObjectRef(2).getCellRef()._signal._numTimesSent, 2);

    // Cell 6 receives only from cell 5 (numTimesSent=4), so numTimesSent should be 4
    EXPECT_EQ(actualData.getObjectRef(6).getCellRef()._signal._numTimesSent, 4);
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
        for (int c = 0; c < NEURONS_PER_CELL; ++c) {
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
    auto param = GetParam();

    // Non-trivial weight and bias values for the test
    constexpr float weight = 0.8f;  // Non-trivial weight (different from default 1.0)
    constexpr float bias = 0.15f;   // Non-trivial bias (different from default 0.0)

    // Setup neural network with non-trivial weight matrix and bias vector:
    // - Channel 'c' uses the specified activation function with custom weight and bias
    // - All other channels use Identity activation with identity weight (1.0) and zero bias
    NeuralNetDesc nn;
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        nn._activationFunctions[i] = (i == param.channelIndex) ? param.activationFunction : ActivationFunction_Identity;
        nn.weight(i, i, (i == param.channelIndex) ? weight : 1.0f);
        nn._biases[i] = (i == param.channelIndex) ? bias : 0.0f;
    }

    nn._connectionWeights[0] = 1.0f;  // Enable input signal reception from connected cell

    // Setup input signal:
    // - Channel 'c' has the specified input value
    // - All other channels have 0 input
    std::vector<float> inputSignal(NEURONS_PER_CELL, 0.0f);
    inputSignal[param.channelIndex] = param.inputValue;

    auto data = Desc()
                    .addCreature({
                        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().neuralNetwork(nn)),
                        ObjectDesc().id(2).pos({0, 1}).type(CellDesc().signal(inputSignal)),
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
    std::vector<float> expected(NEURONS_PER_CELL, 0.0f);

    float preActivation = weight * param.inputValue + bias;
    float rawOutput = applyActivationFunction(param.activationFunction, preActivation);
    expected[param.channelIndex] = std::clamp(rawOutput, -2.0f, 2.0f);

    auto& actual = actualData.getObjectRef(1).getCellRef()._signal._channels;

    constexpr float precision = 0.1f;
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        EXPECT_TRUE(approxCompare(expected[i], actual[i], precision))
            << "Mismatch at channel " << i << ": expected=" << expected[i] << ", actual=" << actual[i];
    }
}

// Test that signals are truncated to [-2, 2] after neural network application
TEST_F(NeuronTests, truncateSignal)
{
    NeuralNetDesc nn;
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        nn.weight(i, i, 2.0f);
    }
    nn._connectionWeights[0] = 1.0f;  // Enable signal forwarding from first connection

    // Input signal: {1.5f, 0, 0, -1.5f, 0, 0, 0, 1.7f, 0, 0, 0, 0, 0, 1.6f, 0, 0}
    // With weight 2.0f on diagonal, outputs are:
    // Channel 0: 1.5 * 2 = 3.0 -> truncated to 2
    // Channel 3: -1.5 * 2 = -3.0 -> truncated to -2
    // Channel 7: 1.7 * 2 = 3.4 -> truncated to 2
    // Channel 13: 1.6 * 2 = 3.2 -> truncated to 2
    Desc data;
    data.addCreature({
        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().neuralNetwork(nn)),
        ObjectDesc().id(2).pos({0, 1}).type(CellDesc().signal({1.5f, 0, 0, -1.5f, 0, 0, 0, 1.7f, 0, 0, 0, 0, 0, 1.6f, 0, 0})),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(std::vector<float>{2, 0, 0, -2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0}, actualData.getObjectRef(1).getCellRef()._signal._channels));
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
    Desc data;
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
