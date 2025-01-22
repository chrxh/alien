#include <cmath>
#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
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
};

TEST_F(NeuronTests, bias)
{
    NeuralNetworkDescription nn;
    nn._biases = {0, 0, 1, 0, 0, 0, 0, -1};

    auto data = DataDescription().addCells({CellDescription().id(1).neuralNetwork(nn)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    EXPECT_TRUE(approxCompare({0, 0, scaledSigmoid(1), 0, 0, 0, 0, scaledSigmoid(-1)}, actualCellById.at(1)._signal->_channels));
}

TEST_F(NeuronTests, weight)
{
    NeuralNetworkDescription nn;
    nn.weight(2, 3, 1.0f);
    nn.weight(2, 7, 0.5f);
    nn.weight(5, 3, -3.5f);

    SignalDescription signal;
    signal._channels = {0, 0, 0, 1, 0, 0, 0, 0.5f};

    auto data = DataDescription().addCells({
        CellDescription()
            .id(1)
            .pos({1.0f, 1.0f})
            .cellType(OscillatorDescription())
            .signal(signal),
        CellDescription().id(2).pos({2.0f, 1.0f}).neuralNetwork(nn),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    EXPECT_TRUE(approxCompare({0, 0, scaledSigmoid(1.0f + 0.5f * 0.5f), 0, 0, scaledSigmoid(-3.5f), 0, 0}, actualCellById.at(2)._signal->_channels));
}

TEST_F(NeuronTests, activationFunctionBinaryStep)
{
    NeuralNetworkDescription nn;
    nn.weight(2, 3, 1.0f);
    nn.weight(2, 7, 0.5f);
    nn.weight(5, 3, -3.5f);
    nn._activationFunctions[2] = ActivationFunction_BinaryStep;
    nn._activationFunctions[5] = ActivationFunction_BinaryStep;


    SignalDescription signal;
    signal._channels = {0, 0, 0, 1, 0, 0, 0, 0.5f};

    auto data = DataDescription().addCells({
        CellDescription()
            .id(1)
            .pos({1.0f, 1.0f})
            .cellType(OscillatorDescription())
            .signal(signal),
        CellDescription().id(2).pos({2.0f, 1.0f}).neuralNetwork(nn),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    EXPECT_TRUE(approxCompare({0, 0, binaryStep(1.0f + 0.5f * 0.5f), 0, 0, binaryStep(-3.5f), 0, 0}, actualCellById.at(2)._signal->_channels));
}
