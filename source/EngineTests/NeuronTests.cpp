#include <cmath>

#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
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
    float scaledSigmoid(float value) const { return 2.0f / (1.0f + std::exp(-value)) - 1.0f; }
    float binaryStep(float value) const { return value >= NEAR_ZERO ? 1.0f : 0.0f; }
    std::vector<float> applyActivationFunction(ActivationFunction af, std::vector<float> const& values)
    {
        std::vector<float> result;
        for (auto const& value : values) {
            switch (af) {
            case ActivationFunction_Sigmoid:
                result.emplace_back(scaledSigmoid(value));
                break;
            case ActivationFunction_BinaryStep:
                result.emplace_back(binaryStep(value));
                break;
            case ActivationFunction_Identity:
                result.emplace_back(value);
                break;
            case ActivationFunction_Abs:
                result.emplace_back(std::abs(value));
                break;
            case ActivationFunction_Gaussian:
                result.emplace_back(expf(-2 * value * value));
                break;
            default:
                THROW_NOT_IMPLEMENTED();
            }
        }
        return result;
    }
};

class NeuronTests_AllActivationFunctions
    : public NeuronTests
    , public testing::WithParamInterface<ActivationFunction>
{};

INSTANTIATE_TEST_SUITE_P(
    NeuronTests_AllActivationFunctions,
    NeuronTests_AllActivationFunctions,
    ::testing::Values(
        ActivationFunction_Sigmoid,
        ActivationFunction_BinaryStep,
        ActivationFunction_Identity,
        ActivationFunction_Abs,
        ActivationFunction_Gaussian));

TEST_P(NeuronTests_AllActivationFunctions, weights)
{
    auto activationFunction = GetParam();

    NeuralNetworkDescription nn;
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        nn._activationFunctions[i] = activationFunction;
        for (int j = 0; j < MAX_CHANNELS; ++j) {
            nn.weight(i, j, 0.0f);
        }
    }
    nn.weight(2, 3, 1.0f);
    nn.weight(2, 7, 0.5f);
    nn.weight(5, 3, -1.5f);

    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({0, 0}).type(CellDescription().neuralNetwork(nn)),
        ObjectDescription().id(2).pos({0, 1}).type(CellDescription().signalAndState({0, 0, 0, 1, 0, 0, 0, 0.5f})),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(
        approxCompare(applyActivationFunction(activationFunction, {0, 0, 1.0f + 0.5f * 0.5f, 0, 0, -1.5f, 0, 0}), actualData.getObjectRef(1).getCellRef()._signal._channels));
}

TEST_P(NeuronTests_AllActivationFunctions, bias)
{
    auto activationFunction = GetParam();

    NeuralNetworkDescription nn;
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        nn._activationFunctions[i] = activationFunction;
        for (int j = 0; j < MAX_CHANNELS; ++j) {
            nn.weight(i, j, 0.0f);
        }
    }
    nn._biases = {0, 0, 1, 0, 0, 0, 0, -1};

    Description data;
    data._objects = {
        ObjectDescription().id(1).pos({0, 0}).type(CellDescription().neuralNetwork(nn)),
        ObjectDescription().id(2).pos({0, 1}).type(CellDescription().signalAndState({0, 0, 0, 0, 0, 0, 0, 0})),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(applyActivationFunction(activationFunction, {0, 0, 1, 0, 0, 0, 0, -1}), actualData.getObjectRef(1).getCellRef()._signal._channels));
}
