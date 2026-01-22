#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ranges>

#include <boost/range/combine.hpp>

#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include <EngineTestData/DescriptionTestDataFactory.h>

#include "IntegrationTestFramework.h"

class MutationTests : public IntegrationTestFramework
{
public:
    MutationTests()
        : IntegrationTestFramework()
    {
        _parameters.mutationNeuralNetwork.value = 1.0f;
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~MutationTests() = default;

protected:
    GenomeDesc createTestGenome() const
    {
        auto const& factory = DescriptionTestDataFactory::get();
        auto nodeParameters = factory.getAllNodeParameters();

        auto gene1 = GeneDesc().nodes({
            factory.createNonDefaultNodeDesc(nodeParameters.at(0)),
            factory.createNonDefaultNodeDesc(nodeParameters.at(1)),
            factory.createNonDefaultNodeDesc(nodeParameters.at(2)),
        });
        auto gene2 = GeneDesc().nodes({
            factory.createNonDefaultNodeDesc(nodeParameters.at(10)),
            factory.createNonDefaultNodeDesc(nodeParameters.at(16)),
            factory.createNonDefaultNodeDesc(nodeParameters.at(22)),
        });

        return GenomeDesc().genes({gene1, gene2});
    }

    bool compareExceptNeuralNetwork(GenomeDesc expected, GenomeDesc actual)
    {
        // compare genomes except neural network, good strategy: set all neural network properties of `expected` and `actual` to 0, then compare both
        auto resetNeuralNetwork = [](GenomeDesc& genome) {
            for (auto& gene : genome._genes) {
                for (auto& node : gene._nodes) {
                    std::fill(node._neuralNetwork._weights.begin(), node._neuralNetwork._weights.end(), 0.0f);
                    std::fill(node._neuralNetwork._biases.begin(), node._neuralNetwork._biases.end(), 0.0f);
                    std::fill(node._neuralNetwork._activationFunctions.begin(), node._neuralNetwork._activationFunctions.end(), ActivationFunction{0});
                }
            }
        };

        resetNeuralNetwork(expected);
        resetNeuralNetwork(actual);

        return expected == actual;
    }
};

TEST_F(MutationTests, neuralNetworkMutation)
{
    auto genome = createTestGenome();

    auto data = Desc().addCreature({ObjectDesc().id(1).type(CellDesc())});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::NeuralNetwork);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // Mutated genome must be equal to original genome except the neural network properties
    EXPECT_TRUE(compareExceptNeuralNetwork(genome, actualGenome));
}
