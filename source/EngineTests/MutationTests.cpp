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
    genome.neuronMutationRate1(NeuronMutationRateDesc().probability(1.0f).weightSigma(1.0f))
        .neuronMutationRate2(NeuronMutationRateDesc().probability(1.0f).weightSigma(1.0f));

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

TEST_F(MutationTests, neuronWeightMutation_weightsActuallyChange)
{
    auto genome = createTestGenome();
    genome.neuronMutationRate1(NeuronMutationRateDesc().probability(1.0f).weightSigma(1.0f))
        .neuronMutationRate2(NeuronMutationRateDesc().probability(0.0f).weightSigma(0.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1).type(CellDesc())}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // At least some weights should have changed
    bool anyWeightChanged = false;
    for (size_t g = 0; g < genome._genes.size() && !anyWeightChanged; ++g) {
        for (size_t n = 0; n < genome._genes.at(g)._nodes.size() && !anyWeightChanged; ++n) {
            auto const& origWeights = genome._genes.at(g)._nodes.at(n)._neuralNetwork._weights;
            auto const& actualWeights = actualGenome._genes.at(g)._nodes.at(n)._neuralNetwork._weights;
            for (size_t w = 0; w < origWeights.size(); ++w) {
                if (origWeights.at(w) != actualWeights.at(w)) {
                    anyWeightChanged = true;
                    break;
                }
            }
        }
    }
    EXPECT_TRUE(anyWeightChanged);
}

TEST_F(MutationTests, neuronWeightMutation_weightsStayClamped)
{
    auto genome = createTestGenome();
    genome.neuronMutationRate1(NeuronMutationRateDesc().probability(1.0f).weightSigma(10.0f))
        .neuronMutationRate2(NeuronMutationRateDesc().probability(1.0f).weightSigma(10.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1).type(CellDesc())}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // All weights must be within [-2, 2] (accounting for NeuralNetWeight quantization)
    for (auto const& gene : actualGenome._genes) {
        for (auto const& node : gene._nodes) {
            for (auto const& w : node._neuralNetwork._weights) {
                EXPECT_GE(w.getValue(), -2.0f - 0.1f);
                EXPECT_LE(w.getValue(), 2.0f + 0.1f);
            }
        }
    }
}

TEST_F(MutationTests, neuronWeightMutation_zeroProbabilityNoChange)
{
    auto genome = createTestGenome();
    genome.neuronMutationRate1(NeuronMutationRateDesc().probability(0.0f).weightSigma(1.0f))
        .neuronMutationRate2(NeuronMutationRateDesc().probability(0.0f).weightSigma(1.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1).type(CellDesc())}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // No weights should have changed
    for (size_t g = 0; g < genome._genes.size(); ++g) {
        for (size_t n = 0; n < genome._genes.at(g)._nodes.size(); ++n) {
            auto const& origWeights = genome._genes.at(g)._nodes.at(n)._neuralNetwork._weights;
            auto const& actualWeights = actualGenome._genes.at(g)._nodes.at(n)._neuralNetwork._weights;
            EXPECT_EQ(origWeights, actualWeights);
        }
    }
}
