#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ranges>

#include <boost/range/combine.hpp>

#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include <EngineTestData/DescriptionTestDataFactory.h>

#include "IntegrationTestFramework.h"

class MutationTests : public IntegrationTestFramework
{
public:
    MutationTests()
        : IntegrationTestFramework()
    {}

    ~MutationTests() = default;

protected:
    GenomeDesc createTestGenome() const
    {
        auto const& factory = DescriptionTestDataFactory::get();
        auto nodeParameters = factory.getAllNodeParameters();

        std::vector<GeneDesc> genes;
        size_t constexpr nodesPerGene = 4;
        for (size_t index = 0; index < nodeParameters.size(); index += nodesPerGene) {
            auto nodeCount = std::min(nodesPerGene, nodeParameters.size() - index);
            std::vector<NodeDesc> nodes;
            nodes.reserve(nodeCount);
            for (size_t offset = 0; offset < nodeCount; ++offset) {
                nodes.push_back(factory.createNonDefaultNodeDesc(nodeParameters.at(index + offset)));
            }
            genes.emplace_back(GeneDesc().nodes(nodes));
        }

        return GenomeDesc().genes(genes);
    }

    bool compareAllExceptNeuronWeights(GenomeDesc expected, GenomeDesc actual)
    {
        auto resetNeuralNet = [](GenomeDesc& genome) {
            for (auto& gene : genome._genes) {
                for (auto& node : gene._nodes) {
                    std::fill(node._neuralNetwork._weights.begin(), node._neuralNetwork._weights.end(), 0.0f);
                }
            }
        };

        resetNeuralNet(expected);
        resetNeuralNet(actual);

        return expected == actual;
    }
};

TEST_F(MutationTests, neuronWeightMutation_keepOtherAttributesUnchanged)
{
    auto genome = createTestGenome();
    genome.neuronWeightMutationRate1(NeuronWeightMutationRateDesc().probability(1.0f).sigma(1.0f))
        .neuronWeightMutationRate2(NeuronWeightMutationRateDesc().probability(1.0f).sigma(1.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1).type(CellDesc())}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // Mutated genome must be equal to original genome except the neural network properties
    EXPECT_TRUE(compareAllExceptNeuronWeights(genome, actualGenome));
}
