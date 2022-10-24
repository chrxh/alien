#include <cmath>
#include <boost/range/combine.hpp>
#include <gtest/gtest.h>

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class NeuronTests : public IntegrationTestFramework
{
public:
    NeuronTests()
        : IntegrationTestFramework({1000, 1000})
    {}

    ~NeuronTests() = default;

protected:
    void expectApproxEqual(std::vector<float> const& expected, std::vector<float> const& actual) const
    {
        CHECK(expected.size() == actual.size())
        for (auto const& [expectedElement, actualElement] : boost::combine(expected, actual)) {
            EXPECT_TRUE(std::abs(expectedElement - actualElement) < 0.01f);
        }
    }

    float scaledSigmoid(float value) const { return 2.0f / (1.0f + std::expf(-value)) - 1.0f; }
};

TEST_F(NeuronTests, bias)
{
    NeuronDescription neuron;
    neuron.bias = {0, 0, 1, 0, 0, 0, 0, -1};

    auto data = DataDescription().addCells({CellDescription().setId(1).setCellFunction(neuron).setMaxConnections(2).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    expectApproxEqual({0, 0, scaledSigmoid(1), 0, 0, 0, 0, scaledSigmoid(-1)}, actualCellById.at(1).activity.channels);
}

TEST_F(NeuronTests, weight)
{
    NeuronDescription neuron;
    neuron.weights[2][3] = 1;

    ActivityDescription activity;
    activity.channels = {0, 0, 0, 1, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription()
            .setId(1)
            .setPos({1.0f, 1.0f})
            .setCellFunction(NerveDescription())
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setActivity(activity),
        CellDescription().setId(2).setPos({2.0f, 1.0f}).setCellFunction(neuron).setMaxConnections(2).setExecutionOrderNumber(0),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    expectApproxEqual({0, 0, scaledSigmoid(1), 0, 0, 0, 0, 0}, actualCellById.at(2).activity.channels);
}
