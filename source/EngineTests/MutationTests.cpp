#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <ranges>
#include <type_traits>
#include <variant>

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
            genome._lineageId = 0;
            genome._prevLineageId = std::nullopt;
            genome._accumulatedMutations = 0.0f;
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
            genome._lineageId = 0;
            genome._prevLineageId = std::nullopt;
            genome._accumulatedMutations = 0.0f;
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
            genome._lineageId = 0;
            genome._prevLineageId = std::nullopt;
            genome._accumulatedMutations = 0.0f;
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
            genome._lineageId = 0;
            genome._prevLineageId = std::nullopt;
            genome._accumulatedMutations = 0.0f;
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

    template <typename T>
    struct IsVariant : std::false_type
    {};
    template <typename... Ts>
    struct IsVariant<std::variant<Ts...>> : std::true_type
    {};

    // Resets all mutable cell type properties to their defaults while preserving the active cell type and its active mode.
    static void resetCellTypeProperties(CellTypeGenomeDesc& cellType)
    {
        std::visit(
            [](auto& cellTypeData) {
                using CellTypeData = std::decay_t<decltype(cellTypeData)>;
                if constexpr (requires { cellTypeData._mode; }) {
                    using ModeType = std::decay_t<decltype(cellTypeData._mode)>;
                    if constexpr (IsVariant<ModeType>::value) {
                        auto mode = cellTypeData._mode;
                        std::visit([](auto& modeData) { modeData = std::decay_t<decltype(modeData)>{}; }, mode);
                        cellTypeData = CellTypeData{};
                        cellTypeData._mode = mode;
                        return;
                    }
                }
                cellTypeData = CellTypeData{};
            },
            cellType);
    }

    bool compareAllExceptCellTypeProperties(GenomeDesc expected, GenomeDesc actual)
    {
        auto reset = [](GenomeDesc& genome) {
            genome._lineageId = 0;
            genome._prevLineageId = std::nullopt;
            genome._accumulatedMutations = 0.0f;
            for (auto& gene : genome._genes) {
                for (auto& node : gene._nodes) {
                    resetCellTypeProperties(node._cellType);
                }
            }
        };
        reset(expected);
        reset(actual);
        return expected == actual;
    }

    GenomeDesc getMutatedGenome(uint64_t objectId = 1) const
    {
        auto actualData = _simulationFacade->getSimulationData();
        auto actualCell = actualData.getObjectRef(objectId).getCellRef();
        auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
        return actualData.getGenomeRef(actualCreature._genomeId);
    }

    template <typename Predicate>
    std::optional<NodeDesc> findNode(GenomeDesc const& genome, Predicate&& predicate) const
    {
        for (auto const& gene : genome._genes) {
            for (auto const& node : gene._nodes) {
                if (predicate(node)) {
                    return node;
                }
            }
        }
        return std::nullopt;
    }
};

TEST_F(MutationTests, neuronWeightMutation_keepOtherAttributesUnchanged)
{
    auto genome = createTestGenome();
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().eventProbability(1.0f).weightSigma(1.0f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().eventProbability(1.0f).weightSigma(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
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
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().eventProbability(1.0f).weightSigma(1.0f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().eventProbability(0.0f).weightSigma(0.0f);

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
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().eventProbability(0.0f).weightSigma(1.0f).biasSigma(1.0f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().eventProbability(0.0f).weightSigma(1.0f).biasSigma(1.0f);

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
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().eventProbability(1.0f).weightSigma(0.0f).biasSigma(1.0f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().eventProbability(0.0f).weightSigma(0.0f).biasSigma(0.0f);

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
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().eventProbability(1.0f).weightSigma(0.0f).biasSigma(0.0f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().eventProbability(1.0f).weightSigma(0.0f).biasSigma(0.0f);

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
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().eventProbability(1.0f).biasSigma(1.0f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().eventProbability(1.0f).biasSigma(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
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
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().eventProbability(1.0f).activationFunctionProbability(1.0f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().eventProbability(0.0f).activationFunctionProbability(0.0f);

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
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().eventProbability(1.0f).activationFunctionProbability(0.0f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().eventProbability(1.0f).activationFunctionProbability(0.0f);

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
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().eventProbability(1.0f).activationFunctionProbability(1.0f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().eventProbability(1.0f).activationFunctionProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
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
    genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().eventProbability(1.0f).sigma(1.0f);
    genome._mutationRates._connectionMutations[1] = ConnectionMutationDesc().eventProbability(0.0f).sigma(0.0f);

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
    genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().eventProbability(0.0f).sigma(1.0f);
    genome._mutationRates._connectionMutations[1] = ConnectionMutationDesc().eventProbability(0.0f).sigma(1.0f);

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
    genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().eventProbability(1.0f).sigma(1.0f);
    genome._mutationRates._connectionMutations[1] = ConnectionMutationDesc().eventProbability(1.0f).sigma(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_TRUE(compareAllExceptConnectionWeights(genome, actualGenome));
}

TEST_F(MutationTests, cellTypePropertiesMutation_changesScalarBoolAndEnumProperties)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().eventProbability(1.0f).sigma(1.0f).probability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();

    auto originalDigestor = findNode(genome, [](NodeDesc const& node) { return std::holds_alternative<DigestorGenomeDesc>(node._cellType); });
    auto actualDigestor = findNode(actualGenome, [](NodeDesc const& node) { return std::holds_alternative<DigestorGenomeDesc>(node._cellType); });
    ASSERT_TRUE(originalDigestor);
    ASSERT_TRUE(actualDigestor);
    EXPECT_NE(
        std::get<DigestorGenomeDesc>(originalDigestor->_cellType)._rawEnergyConductivity,
        std::get<DigestorGenomeDesc>(actualDigestor->_cellType)._rawEnergyConductivity);

    auto originalMemory = findNode(genome, [](NodeDesc const& node) {
        return std::holds_alternative<MemoryGenomeDesc>(node._cellType)
            && std::holds_alternative<SignalStorageGenomeDesc>(std::get<MemoryGenomeDesc>(node._cellType)._mode);
    });
    auto actualMemory = findNode(actualGenome, [](NodeDesc const& node) {
        return std::holds_alternative<MemoryGenomeDesc>(node._cellType)
            && std::holds_alternative<SignalStorageGenomeDesc>(std::get<MemoryGenomeDesc>(node._cellType)._mode);
    });
    ASSERT_TRUE(originalMemory);
    ASSERT_TRUE(actualMemory);
    EXPECT_NE(
        std::get<SignalStorageGenomeDesc>(std::get<MemoryGenomeDesc>(originalMemory->_cellType)._mode)._readOnly,
        std::get<SignalStorageGenomeDesc>(std::get<MemoryGenomeDesc>(actualMemory->_cellType)._mode)._readOnly);

    auto originalCommunicator = findNode(genome, [](NodeDesc const& node) {
        return std::holds_alternative<CommunicatorGenomeDesc>(node._cellType)
            && std::holds_alternative<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(node._cellType)._mode);
    });
    auto actualCommunicator = findNode(actualGenome, [](NodeDesc const& node) {
        return std::holds_alternative<CommunicatorGenomeDesc>(node._cellType)
            && std::holds_alternative<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(node._cellType)._mode);
    });
    ASSERT_TRUE(originalCommunicator);
    ASSERT_TRUE(actualCommunicator);
    EXPECT_NE(
        std::get<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(originalCommunicator->_cellType)._mode)._restrictToLineage,
        std::get<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(actualCommunicator->_cellType)._mode)._restrictToLineage);
}

TEST_F(MutationTests, cellTypePropertiesMutation_changesBitsetPropertiesBitwise)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().eventProbability(1.0f).probability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();

    auto originalSensor = findNode(genome, [](NodeDesc const& node) {
        return std::holds_alternative<SensorGenomeDesc>(node._cellType)
            && std::holds_alternative<DetectFreeCellGenomeDesc>(std::get<SensorGenomeDesc>(node._cellType)._mode);
    });
    auto actualSensor = findNode(actualGenome, [](NodeDesc const& node) {
        return std::holds_alternative<SensorGenomeDesc>(node._cellType)
            && std::holds_alternative<DetectFreeCellGenomeDesc>(std::get<SensorGenomeDesc>(node._cellType)._mode);
    });
    ASSERT_TRUE(originalSensor);
    ASSERT_TRUE(actualSensor);
    EXPECT_EQ(
        (std::get<DetectFreeCellGenomeDesc>(std::get<SensorGenomeDesc>(originalSensor->_cellType)._mode)._restrictToColors ^ Const::RestrictToColors_Max),
        std::get<DetectFreeCellGenomeDesc>(std::get<SensorGenomeDesc>(actualSensor->_cellType)._mode)._restrictToColors);

    auto originalMemory = findNode(genome, [](NodeDesc const& node) { return std::holds_alternative<MemoryGenomeDesc>(node._cellType); });
    auto actualMemory = findNode(actualGenome, [](NodeDesc const& node) { return std::holds_alternative<MemoryGenomeDesc>(node._cellType); });
    ASSERT_TRUE(originalMemory);
    ASSERT_TRUE(actualMemory);
    EXPECT_EQ(
        static_cast<uint16_t>(std::get<MemoryGenomeDesc>(originalMemory->_cellType)._channelBitMask ^ Const::MemoryChannelBitMask_Max),
        std::get<MemoryGenomeDesc>(actualMemory->_cellType)._channelBitMask);

    auto originalCommunicator = findNode(genome, [](NodeDesc const& node) {
        return std::holds_alternative<CommunicatorGenomeDesc>(node._cellType)
            && std::holds_alternative<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(node._cellType)._mode);
    });
    auto actualCommunicator = findNode(actualGenome, [](NodeDesc const& node) {
        return std::holds_alternative<CommunicatorGenomeDesc>(node._cellType)
            && std::holds_alternative<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(node._cellType)._mode);
    });
    ASSERT_TRUE(originalCommunicator);
    ASSERT_TRUE(actualCommunicator);
    EXPECT_EQ(
        (std::get<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(originalCommunicator->_cellType)._mode)._restrictToColors ^ Const::RestrictToColors_Max),
        std::get<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(actualCommunicator->_cellType)._mode)._restrictToColors);
}

TEST_F(MutationTests, cellTypePropertiesMutation_doesNotChangeOtherAttributes)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().eventProbability(1.0f).sigma(1.0f).probability(1.0f);
    genome._mutationRates._cellTypePropertiesMutations[1] = CellTypePropertiesMutationDesc().eventProbability(1.0f).sigma(1.0f).probability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    EXPECT_TRUE(compareAllExceptCellTypeProperties(genome, actualGenome));
}

TEST_F(MutationTests, metaMutation_neuronRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().eventProbability(0.5f).weightSigma(0.5f).biasSigma(0.5f).activationFunctionProbability(0.5f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().eventProbability(0.5f).weightSigma(0.5f).biasSigma(0.5f).activationFunctionProbability(0.5f);

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

    bool anyChanged = actualGenome._mutationRates._neuronMutations[0]._eventProbability != 0.5f
        || actualGenome._mutationRates._neuronMutations[0]._weightSigma != 0.5f || actualGenome._mutationRates._neuronMutations[0]._biasSigma != 0.5f
        || actualGenome._mutationRates._neuronMutations[0]._activationFunctionProbability != 0.5f
        || actualGenome._mutationRates._neuronMutations[1]._eventProbability != 0.5f || actualGenome._mutationRates._neuronMutations[1]._weightSigma != 0.5f
        || actualGenome._mutationRates._neuronMutations[1]._biasSigma != 0.5f
        || actualGenome._mutationRates._neuronMutations[1]._activationFunctionProbability != 0.5f;
    EXPECT_TRUE(anyChanged);

    EXPECT_GE(actualGenome._mutationRates._neuronMutations[0]._eventProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[0]._weightSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[0]._biasSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[0]._activationFunctionProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[1]._eventProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[1]._weightSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[1]._biasSigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._neuronMutations[1]._activationFunctionProbability, 0.0f);
}

TEST_F(MutationTests, metaMutation_neuronRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._neuronMutations[0] = NeuronMutationDesc().eventProbability(0.5f).weightSigma(0.5f).biasSigma(0.5f).activationFunctionProbability(0.5f);
    genome._mutationRates._neuronMutations[1] = NeuronMutationDesc().eventProbability(0.5f).weightSigma(0.5f).biasSigma(0.5f).activationFunctionProbability(0.5f);

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

    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._eventProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._weightSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._biasSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[0]._activationFunctionProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._eventProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._weightSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._biasSigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._neuronMutations[1]._activationFunctionProbability, 0.5f);
}

TEST_F(MutationTests, metaMutation_connectionRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().eventProbability(0.5f).sigma(0.5f);
    genome._mutationRates._connectionMutations[1] = ConnectionMutationDesc().eventProbability(0.5f).sigma(0.5f);

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

    bool anyChanged = actualGenome._mutationRates._connectionMutations[0]._eventProbability != 0.5f
        || actualGenome._mutationRates._connectionMutations[0]._sigma != 0.5f || actualGenome._mutationRates._connectionMutations[1]._eventProbability != 0.5f
        || actualGenome._mutationRates._connectionMutations[1]._sigma != 0.5f;
    EXPECT_TRUE(anyChanged);
    EXPECT_GE(actualGenome._mutationRates._connectionMutations[0]._eventProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._connectionMutations[0]._sigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._connectionMutations[1]._eventProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._connectionMutations[1]._sigma, 0.0f);
}

TEST_F(MutationTests, metaMutation_connectionRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().eventProbability(0.5f).sigma(0.5f);
    genome._mutationRates._connectionMutations[1] = ConnectionMutationDesc().eventProbability(0.5f).sigma(0.5f);

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

    EXPECT_EQ(actualGenome._mutationRates._connectionMutations[0]._eventProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._connectionMutations[0]._sigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._connectionMutations[1]._eventProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._connectionMutations[1]._sigma, 0.5f);
}

TEST_F(MutationTests, metaMutation_cellTypePropertyRatesActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().eventProbability(0.5f).sigma(0.5f).probability(0.5f);
    genome._mutationRates._cellTypePropertiesMutations[1] = CellTypePropertiesMutationDesc().eventProbability(0.4f).sigma(0.4f).probability(0.4f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationCellTypePropertiesSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    bool anyChanged = actualGenome._mutationRates._cellTypePropertiesMutations[0]._eventProbability != 0.5f
        || actualGenome._mutationRates._cellTypePropertiesMutations[0]._sigma != 0.5f
        || actualGenome._mutationRates._cellTypePropertiesMutations[0]._probability != 0.5f
        || actualGenome._mutationRates._cellTypePropertiesMutations[1]._eventProbability != 0.4f
        || actualGenome._mutationRates._cellTypePropertiesMutations[1]._sigma != 0.4f
        || actualGenome._mutationRates._cellTypePropertiesMutations[1]._probability != 0.4f;
    EXPECT_TRUE(anyChanged);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[0]._eventProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[0]._sigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[0]._probability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[1]._eventProbability, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[1]._sigma, 0.0f);
    EXPECT_GE(actualGenome._mutationRates._cellTypePropertiesMutations[1]._probability, 0.0f);
}

TEST_F(MutationTests, metaMutation_cellTypePropertyRatesZeroSigmaNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().eventProbability(0.5f).sigma(0.5f).probability(0.5f);
    genome._mutationRates._cellTypePropertiesMutations[1] = CellTypePropertiesMutationDesc().eventProbability(0.4f).sigma(0.4f).probability(0.4f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationCellTypePropertiesSigma.value = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[0]._eventProbability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[0]._sigma, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[0]._probability, 0.5f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[1]._eventProbability, 0.4f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[1]._sigma, 0.4f);
    EXPECT_EQ(actualGenome._mutationRates._cellTypePropertiesMutations[1]._probability, 0.4f);
}

enum class MutationType
{
    Neuron,
    Connection,
    CellTypeProperties
};

class MutationsTests_AllTypes
    : public MutationTests
    , public ::testing::WithParamInterface<MutationType>
{
};

INSTANTIATE_TEST_SUITE_P(
    MutationTests,
    MutationsTests_AllTypes,
    ::testing::Values(MutationType::Neuron, MutationType::Connection, MutationType::CellTypeProperties));

TEST_P(MutationsTests_AllTypes, accumulatedMutations_increases)
{
    auto genome = createTestGenome().lineageId(42).prevLineageId(41);
    switch (GetParam()) {
    case MutationType::Neuron:
        genome._mutationRates._neuronMutations[0] =
            NeuronMutationDesc().eventProbability(1.0f).weightSigma(1.0f).biasSigma(1.0f).activationFunctionProbability(1.0f);
        break;
    case MutationType::Connection:
        genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().eventProbability(1.0f).sigma(1.0f);
        break;
    case MutationType::CellTypeProperties:
        genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().eventProbability(1.0f).sigma(1.0f).probability(1.0f);
        break;
    }

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    auto parameters = _parameters;
    parameters.newLineageThreshold.value = 1.0e30f;
    _simulationFacade->setSimulationParameters(parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_GT(actualGenome._accumulatedMutations, genome._accumulatedMutations);
}

TEST_F(MutationTests, accumulatedMutations_metaMutationDoesNotAccount)
{
    auto genome = GenomeDesc().lineageId(42).prevLineageId(41);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.metaMutationNeuronsSigma.value = 1.0f;
    _parameters.metaMutationConnectionsSigma.value = 1.0f;
    _parameters.metaMutationCellTypePropertiesSigma.value = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._accumulatedMutations, 0.0f);
    EXPECT_EQ(actualGenome._lineageId, 42);
    EXPECT_EQ(actualGenome._prevLineageId, 41);
}

TEST_F(MutationTests, accumulatedMutations_createsNewLineageId)
{
    auto genome = createTestGenome().lineageId(42).prevLineageId(41);
    genome._accumulatedMutations = 11.0f;

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _parameters.newLineageThreshold.value = 0.1f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_TRUE(actualGenome._prevLineageId.has_value());
    EXPECT_EQ(*actualGenome._prevLineageId, 42);
    EXPECT_GT(actualGenome._lineageId, 42);
}
