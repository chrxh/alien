#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ranges>

#include <boost/range/combine.hpp>

#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include <EngineTestData/DescTestDataFactory.h>

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
        auto const& factory = DescTestDataFactory::get();
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
        auto reset = [](GenomeDesc& genome) {
            for (auto& gene : genome._genes) {
                for (auto& node : gene._nodes) {
                    std::fill(node._neuralNetwork._weights.begin(), node._neuralNetwork._weights.end(), 0.0f);
                }
            }
        };
        reset(expected);
        reset(actual);
        return expected == actual;
    }

    bool compareAllExceptNeuronBiases(GenomeDesc expected, GenomeDesc actual)
    {
        auto reset = [](GenomeDesc& genome) {
            for (auto& gene : genome._genes) {
                for (auto& node : gene._nodes) {
                    std::fill(node._neuralNetwork._biases.begin(), node._neuralNetwork._biases.end(), 0.0f);
                }
            }
        };
        reset(expected);
        reset(actual);
        return expected == actual;
    }

    bool compareAllExceptActivationFunctions(GenomeDesc expected, GenomeDesc actual)
    {
        auto reset = [](GenomeDesc& genome) {
            for (auto& gene : genome._genes) {
                for (auto& node : gene._nodes) {
                    std::fill(node._neuralNetwork._activationFunctions.begin(), node._neuralNetwork._activationFunctions.end(), ActivationFunction_Identity);
                }
            }
        };
        reset(expected);
        reset(actual);
        return expected == actual;
    }

    bool compareAllExceptConnectionWeights(GenomeDesc expected, GenomeDesc actual)
    {
        auto reset = [](GenomeDesc& genome) {
            for (auto& gene : genome._genes) {
                for (auto& node : gene._nodes) {
                    std::fill(node._neuralNetwork._connectionWeights.begin(), node._neuralNetwork._connectionWeights.end(), 0.0f);
                }
            }
        };
        reset(expected);
        reset(actual);
        return expected == actual;
    }
};

TEST_F(MutationTests, neuronWeightMutation_keepOtherAttributesUnchanged)
{
    auto genome = createTestGenome();
    genome.neuronMutation1(NeuronMutationDesc().probability(1.0f).weightSigma(1.0f))
        .neuronMutation2(NeuronMutationDesc().probability(1.0f).weightSigma(1.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 1000; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_TRUE(compareAllExceptNeuronWeights(genome, actualGenome));
}

TEST_F(MutationTests, neuronWeightMutation_weightsActuallyChange)
{
    auto genome = createTestGenome();
    genome.neuronMutation1(NeuronMutationDesc().probability(1.0f).weightSigma(1.0f))
        .neuronMutation2(NeuronMutationDesc().probability(0.0f).weightSigma(0.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

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
    genome.neuronMutation1(NeuronMutationDesc().probability(0.0f).weightSigma(1.0f).biasSigma(1.0f))
        .neuronMutation2(NeuronMutationDesc().probability(0.0f).weightSigma(1.0f).biasSigma(1.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // No weights or biases should have changed
    for (size_t g = 0; g < genome._genes.size(); ++g) {
        for (size_t n = 0; n < genome._genes.at(g)._nodes.size(); ++n) {
            auto const& origWeights = genome._genes.at(g)._nodes.at(n)._neuralNetwork._weights;
            auto const& actualWeights = actualGenome._genes.at(g)._nodes.at(n)._neuralNetwork._weights;
            EXPECT_EQ(origWeights, actualWeights);
            auto const& origBiases = genome._genes.at(g)._nodes.at(n)._neuralNetwork._biases;
            auto const& actualBiases = actualGenome._genes.at(g)._nodes.at(n)._neuralNetwork._biases;
            EXPECT_EQ(origBiases, actualBiases);
        }
    }
}

TEST_F(MutationTests, neuronBiasMutation_biasesActuallyChange)
{
    auto genome = createTestGenome();
    genome.neuronMutation1(NeuronMutationDesc().probability(1.0f).weightSigma(0.0f).biasSigma(1.0f))
        .neuronMutation2(NeuronMutationDesc().probability(0.0f).weightSigma(0.0f).biasSigma(0.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // At least some biases should have changed
    bool anyBiasChanged = false;
    for (size_t g = 0; g < genome._genes.size() && !anyBiasChanged; ++g) {
        for (size_t n = 0; n < genome._genes.at(g)._nodes.size() && !anyBiasChanged; ++n) {
            auto const& origBiases = genome._genes.at(g)._nodes.at(n)._neuralNetwork._biases;
            auto const& actualBiases = actualGenome._genes.at(g)._nodes.at(n)._neuralNetwork._biases;
            for (size_t b = 0; b < origBiases.size(); ++b) {
                if (origBiases.at(b) != actualBiases.at(b)) {
                    anyBiasChanged = true;
                    break;
                }
            }
        }
    }
    EXPECT_TRUE(anyBiasChanged);

    // All biases must be within [-2, 2]
    for (auto const& gene : actualGenome._genes) {
        for (auto const& node : gene._nodes) {
            for (auto const& b : node._neuralNetwork._biases) {
                EXPECT_GE(b, -2.0f - 0.1f);
                EXPECT_LE(b, 2.0f + 0.1f);
            }
        }
    }
}

TEST_F(MutationTests, neuronBiasMutation_zeroBiasSigmaNoChange)
{
    auto genome = createTestGenome();
    genome.neuronMutation1(NeuronMutationDesc().probability(1.0f).weightSigma(0.0f).biasSigma(0.0f))
        .neuronMutation2(NeuronMutationDesc().probability(1.0f).weightSigma(0.0f).biasSigma(0.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // No biases should have changed
    for (size_t g = 0; g < genome._genes.size(); ++g) {
        for (size_t n = 0; n < genome._genes.at(g)._nodes.size(); ++n) {
            auto const& origBiases = genome._genes.at(g)._nodes.at(n)._neuralNetwork._biases;
            auto const& actualBiases = actualGenome._genes.at(g)._nodes.at(n)._neuralNetwork._biases;
            EXPECT_EQ(origBiases, actualBiases);
        }
    }
}

TEST_F(MutationTests, neuronBiasMutation_keepOtherAttributesUnchanged)
{
    auto genome = createTestGenome();
    genome.neuronMutation1(NeuronMutationDesc().probability(1.0f).biasSigma(1.0f))
        .neuronMutation2(NeuronMutationDesc().probability(1.0f).biasSigma(1.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 1000; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_TRUE(compareAllExceptNeuronBiases(genome, actualGenome));
}

TEST_F(MutationTests, neuronActivationFunctionMutation_activationFunctionsActuallyChange)
{
    auto genome = createTestGenome();
    genome.neuronMutation1(NeuronMutationDesc().probability(1.0f).activationFunctionProbability(1.0f))
        .neuronMutation2(NeuronMutationDesc().probability(0.0f).activationFunctionProbability(0.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // At least some activation functions should have changed
    bool anyActivationFunctionChanged = false;
    for (size_t g = 0; g < genome._genes.size() && !anyActivationFunctionChanged; ++g) {
        for (size_t n = 0; n < genome._genes.at(g)._nodes.size() && !anyActivationFunctionChanged; ++n) {
            auto const& origFunctions = genome._genes.at(g)._nodes.at(n)._neuralNetwork._activationFunctions;
            auto const& actualFunctions = actualGenome._genes.at(g)._nodes.at(n)._neuralNetwork._activationFunctions;
            for (size_t f = 0; f < origFunctions.size(); ++f) {
                if (origFunctions.at(f) != actualFunctions.at(f)) {
                    anyActivationFunctionChanged = true;
                    break;
                }
            }
        }
    }
    EXPECT_TRUE(anyActivationFunctionChanged);

    // All activation functions must be valid
    for (auto const& gene : actualGenome._genes) {
        for (auto const& node : gene._nodes) {
            for (auto const& f : node._neuralNetwork._activationFunctions) {
                EXPECT_GE(f, 0);
                EXPECT_LT(f, ActivationFunction_Count);
            }
        }
    }
}

TEST_F(MutationTests, neuronActivationFunctionMutation_zeroProbabilityNoChange)
{
    auto genome = createTestGenome();
    genome.neuronMutation1(NeuronMutationDesc().probability(1.0f).activationFunctionProbability(0.0f))
        .neuronMutation2(NeuronMutationDesc().probability(1.0f).activationFunctionProbability(0.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // No activation functions should have changed
    for (size_t g = 0; g < genome._genes.size(); ++g) {
        for (size_t n = 0; n < genome._genes.at(g)._nodes.size(); ++n) {
            auto const& origFunctions = genome._genes.at(g)._nodes.at(n)._neuralNetwork._activationFunctions;
            auto const& actualFunctions = actualGenome._genes.at(g)._nodes.at(n)._neuralNetwork._activationFunctions;
            EXPECT_EQ(origFunctions, actualFunctions);
        }
    }
}

TEST_F(MutationTests, neuronActivationFunctionMutation_keepOtherAttributesUnchanged)
{
    auto genome = createTestGenome();
    genome.neuronMutation1(NeuronMutationDesc().probability(1.0f).activationFunctionProbability(1.0f))
        .neuronMutation2(NeuronMutationDesc().probability(1.0f).activationFunctionProbability(1.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 1000; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_TRUE(compareAllExceptActivationFunctions(genome, actualGenome));
}

TEST_F(MutationTests, connectionWeightMutation_weightsActuallyChange)
{
    auto genome = createTestGenome();
    genome.connectionMutationRate1(ConnectionMutationDesc().probability(1.0f).sigma(1.0f))
        .connectionMutationRate2(ConnectionMutationDesc().probability(0.0f).sigma(0.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // At least some connection weights should have changed
    bool anyWeightChanged = false;
    for (size_t g = 0; g < genome._genes.size() && !anyWeightChanged; ++g) {
        for (size_t n = 0; n < genome._genes.at(g)._nodes.size() && !anyWeightChanged; ++n) {
            auto const& origWeights = genome._genes.at(g)._nodes.at(n)._neuralNetwork._connectionWeights;
            auto const& actualWeights = actualGenome._genes.at(g)._nodes.at(n)._neuralNetwork._connectionWeights;
            for (size_t w = 0; w < origWeights.size(); ++w) {
                if (origWeights.at(w) != actualWeights.at(w)) {
                    anyWeightChanged = true;
                    break;
                }
            }
        }
    }
    EXPECT_TRUE(anyWeightChanged);

    // All connection weights must be within [-2, 2]
    for (auto const& gene : actualGenome._genes) {
        for (auto const& node : gene._nodes) {
            for (auto const& w : node._neuralNetwork._connectionWeights) {
                EXPECT_GE(w, -2.0f - 0.1f);
                EXPECT_LE(w, 2.0f + 0.1f);
            }
        }
    }
}

TEST_F(MutationTests, connectionWeightMutation_zeroProbabilityNoChange)
{
    auto genome = createTestGenome();
    genome.connectionMutationRate1(ConnectionMutationDesc().probability(0.0f).sigma(1.0f))
        .connectionMutationRate2(ConnectionMutationDesc().probability(0.0f).sigma(1.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // No connection weights should have changed
    for (size_t g = 0; g < genome._genes.size(); ++g) {
        for (size_t n = 0; n < genome._genes.at(g)._nodes.size(); ++n) {
            auto const& origWeights = genome._genes.at(g)._nodes.at(n)._neuralNetwork._connectionWeights;
            auto const& actualWeights = actualGenome._genes.at(g)._nodes.at(n)._neuralNetwork._connectionWeights;
            EXPECT_EQ(origWeights, actualWeights);
        }
    }
}

TEST_F(MutationTests, connectionWeightMutation_keepOtherAttributesUnchanged)
{
    auto genome = createTestGenome();
    genome.connectionMutationRate1(ConnectionMutationDesc().probability(1.0f).sigma(1.0f))
        .connectionMutationRate2(ConnectionMutationDesc().probability(1.0f).sigma(1.0f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 1000; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_TRUE(compareAllExceptConnectionWeights(genome, actualGenome));
}

TEST_F(MutationTests, lineageMutation_lineageIdActuallyChanges)
{
    auto genome = createTestGenome();
    genome.lineageId(42).lineageMutationProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_NE(actualGenome._lineageId, 42);
}

TEST_F(MutationTests, lineageMutation_zeroProbabilityNoChange)
{
    auto genome = createTestGenome();
    genome.lineageId(42).lineageMutationProbability(0.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_EQ(actualGenome._lineageId, 42);
}

TEST_F(MutationTests, lineageMutation_keepOtherAttributesUnchanged)
{
    auto genome = createTestGenome();
    genome.lineageId(42).lineageMutationProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // Reset lineageId on both to compare everything else
    genome._lineageId = 0;
    genome._prevLineageId = 0;
    actualGenome._lineageId = 0;
    actualGenome._prevLineageId = 0;
    EXPECT_EQ(genome, actualGenome);
}

TEST_F(MutationTests, metaMutation_neuronRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome.neuronMutation1(NeuronMutationDesc().probability(0.5f).weightSigma(0.5f).biasSigma(0.5f).activationFunctionProbability(0.5f))
        .neuronMutation2(NeuronMutationDesc().probability(0.5f).weightSigma(0.5f).biasSigma(0.5f).activationFunctionProbability(0.5f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationNeuronsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    bool anyChanged = actualGenome._neuronMutation1._probability != 0.5f || actualGenome._neuronMutation1._weightSigma != 0.5f
        || actualGenome._neuronMutation1._biasSigma != 0.5f || actualGenome._neuronMutation1._activationFunctionProbability != 0.5f
        || actualGenome._neuronMutation2._probability != 0.5f || actualGenome._neuronMutation2._weightSigma != 0.5f
        || actualGenome._neuronMutation2._biasSigma != 0.5f || actualGenome._neuronMutation2._activationFunctionProbability != 0.5f;
    EXPECT_TRUE(anyChanged);

    EXPECT_GE(actualGenome._neuronMutation1._probability, 0.0f);
    EXPECT_GE(actualGenome._neuronMutation1._weightSigma, 0.0f);
    EXPECT_GE(actualGenome._neuronMutation1._biasSigma, 0.0f);
    EXPECT_GE(actualGenome._neuronMutation1._activationFunctionProbability, 0.0f);
    EXPECT_GE(actualGenome._neuronMutation2._probability, 0.0f);
    EXPECT_GE(actualGenome._neuronMutation2._weightSigma, 0.0f);
    EXPECT_GE(actualGenome._neuronMutation2._biasSigma, 0.0f);
    EXPECT_GE(actualGenome._neuronMutation2._activationFunctionProbability, 0.0f);
}

TEST_F(MutationTests, metaMutation_neuronRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome.neuronMutation1(NeuronMutationDesc().probability(0.5f).weightSigma(0.5f).biasSigma(0.5f).activationFunctionProbability(0.5f))
        .neuronMutation2(NeuronMutationDesc().probability(0.5f).weightSigma(0.5f).biasSigma(0.5f).activationFunctionProbability(0.5f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationNeuronsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_EQ(actualGenome._neuronMutation1._probability, 0.5f);
    EXPECT_EQ(actualGenome._neuronMutation1._weightSigma, 0.5f);
    EXPECT_EQ(actualGenome._neuronMutation1._biasSigma, 0.5f);
    EXPECT_EQ(actualGenome._neuronMutation1._activationFunctionProbability, 0.5f);
    EXPECT_EQ(actualGenome._neuronMutation2._probability, 0.5f);
    EXPECT_EQ(actualGenome._neuronMutation2._weightSigma, 0.5f);
    EXPECT_EQ(actualGenome._neuronMutation2._biasSigma, 0.5f);
    EXPECT_EQ(actualGenome._neuronMutation2._activationFunctionProbability, 0.5f);
}

TEST_F(MutationTests, metaMutation_connectionRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome.connectionMutationRate1(ConnectionMutationDesc().probability(0.5f).sigma(0.5f))
        .connectionMutationRate2(ConnectionMutationDesc().probability(0.5f).sigma(0.5f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationConnectionsSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    bool anyChanged = actualGenome._connectionMutationRate1._probability != 0.5f || actualGenome._connectionMutationRate1._sigma != 0.5f
        || actualGenome._connectionMutationRate2._probability != 0.5f || actualGenome._connectionMutationRate2._sigma != 0.5f;
    EXPECT_TRUE(anyChanged);
    EXPECT_GE(actualGenome._connectionMutationRate1._probability, 0.0f);
    EXPECT_GE(actualGenome._connectionMutationRate1._sigma, 0.0f);
    EXPECT_GE(actualGenome._connectionMutationRate2._probability, 0.0f);
    EXPECT_GE(actualGenome._connectionMutationRate2._sigma, 0.0f);
}

TEST_F(MutationTests, metaMutation_connectionRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome.connectionMutationRate1(ConnectionMutationDesc().probability(0.5f).sigma(0.5f))
        .connectionMutationRate2(ConnectionMutationDesc().probability(0.5f).sigma(0.5f));

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationConnectionsSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_EQ(actualGenome._connectionMutationRate1._probability, 0.5f);
    EXPECT_EQ(actualGenome._connectionMutationRate1._sigma, 0.5f);
    EXPECT_EQ(actualGenome._connectionMutationRate2._probability, 0.5f);
    EXPECT_EQ(actualGenome._connectionMutationRate2._sigma, 0.5f);
}

TEST_F(MutationTests, metaMutation_lineageProbabilityActuallyChanges)
{
    auto genome = createTestGenome();
    genome.lineageMutationProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationLineagesSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_NE(actualGenome._lineageMutationProbability, 0.5f);
}

TEST_F(MutationTests, metaMutation_lineageProbabilityZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome.lineageMutationProbability(0.5f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationLineagesSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_EQ(actualGenome._lineageMutationProbability, 0.5f);
    EXPECT_GE(actualGenome._lineageMutationProbability, 0.0f);
}
