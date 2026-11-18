#include <optional>

#include <gtest/gtest.h>

#include <Data/CellTypeConstants.h>
#include <Data/DescValidationService.h>
#include <Data/Descs.h>

#include <Engine/Interface/SimulationFacade.h>

#include "MutationTestsBase.h"

namespace
{
    size_t constexpr NumGenes = 400;
    auto constexpr OldNodeAngle = 45.0f;
}

enum class NewNodeSource
{
    ExtendGene,
    AddNode,
    AddGene
};

class NewNodeMutationTests
    : public MutationTestsBase
    , public ::testing::WithParamInterface<NewNodeSource>
{
protected:
    GenomeDesc createGenomeWithNewNodes()
    {
        auto genome = GenomeDesc().genes(std::vector<GeneDesc>(NumGenes, GeneDesc().nodes({NodeDesc().referenceAngle(OldNodeAngle)})));
        switch (GetParam()) {
        case NewNodeSource::ExtendGene:
            genome._mutationRates._extendGeneMutation = ExtendGeneMutationDesc().geneProbability(1.0f);
            break;
        case NewNodeSource::AddNode:
            genome._mutationRates._addNodeMutation = AddNodeMutationDesc().nodeProbability(1.0f);
            break;
        case NewNodeSource::AddGene:
            genome._mutationRates._addGeneMutation = AddGeneMutationDesc().geneProbability(1.0f);
            break;
        }

        _simulationFacade->setSimulationData(ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome));
        _simulationFacade->testOnly_mutate(1);
        return getMutatedGenome();
    }

    static std::vector<NodeDesc> getNewNodes(GenomeDesc const& genome)
    {
        std::vector<NodeDesc> result;
        for (auto const& gene : genome._genes) {
            for (auto const& node : gene._nodes) {
                if (node._referenceAngle != OldNodeAngle) {
                    result.emplace_back(node);
                }
            }
        }
        return result;
    }

    static bool hasDefaultNeuralNetwork(NodeDesc const& node) { return node._neuralNetwork == NeuralNetGenomeDesc(); }

    // Returns no value for cell types without a mode.
    static std::optional<bool> hasDefaultCellTypeMode(NodeDesc const& node)
    {
        return std::visit(
            [](auto const& cellTypeData) -> std::optional<bool> {
                if constexpr (requires { cellTypeData._mode; }) {
                    auto defaultMode = std::decay_t<decltype(cellTypeData)>()._mode;
                    if constexpr (IsVariant<decltype(defaultMode)>::value) {
                        return cellTypeData._mode.index() == defaultMode.index();
                    } else {
                        return cellTypeData._mode == defaultMode;
                    }
                } else {
                    return std::nullopt;
                }
            },
            node._cellType);
    }
};

INSTANTIATE_TEST_SUITE_P(
    NewNodeMutationTests,
    NewNodeMutationTests,
    ::testing::Values(NewNodeSource::ExtendGene, NewNodeSource::AddNode, NewNodeSource::AddGene));

TEST_P(NewNodeMutationTests, newNode_halfKeepDefaultNeuralNetwork)
{
    auto newNodes = getNewNodes(createGenomeWithNewNodes());
    ASSERT_LE(NumGenes, newNodes.size());

    auto numDefaultNodes = std::ranges::count_if(newNodes, hasDefaultNeuralNetwork);
    auto defaultFraction = static_cast<float>(numDefaultNodes) / static_cast<float>(newNodes.size());
    EXPECT_LT(0.35f, defaultFraction);
    EXPECT_GT(0.65f, defaultFraction);
}

TEST_P(NewNodeMutationTests, newNode_randomNeuralNetworkIsSparse)
{
    std::ptrdiff_t numRandomNodes = 0;
    std::ptrdiff_t numWeights = 0;
    std::ptrdiff_t numBiases = 0;
    std::ptrdiff_t numActivationFunctions = 0;
    for (auto const& node : getNewNodes(createGenomeWithNewNodes())) {
        if (hasDefaultNeuralNetwork(node)) {
            continue;
        }
        auto const& neuralNetwork = node._neuralNetwork;
        ++numRandomNodes;
        numWeights += std::ranges::count_if(neuralNetwork._weights, [](auto const& weight) { return weight.getValue() != 0.0f; });
        numBiases += std::ranges::count_if(neuralNetwork._biases, [](auto const& bias) { return bias != 0.0f; });
        numActivationFunctions +=
            std::ranges::count_if(neuralNetwork._activationFunctions, [](auto const& function) { return function != ActivationFunction_Identity; });
    }

    // Random weights, biases and activation functions occur, but fewer than 1 in 8 values differs from 0 or Identity.
    auto expectSparse = [&](std::ptrdiff_t numRandomValues, std::ptrdiff_t numValuesPerNode) {
        EXPECT_LT(0, numRandomValues);
        EXPECT_GT(numRandomNodes * numValuesPerNode / 8, numRandomValues);
    };
    expectSparse(numWeights, NEURAL_NET_OUTPUTS * NEURAL_NET_INPUTS);
    expectSparse(numBiases, NEURAL_NET_OUTPUTS);
    expectSparse(numActivationFunctions, NEURAL_NET_OUTPUTS);
}

TEST_P(NewNodeMutationTests, newNode_drawsCellTypeModeFromAllModes)
{
    auto numNodesWithMode = 0;
    auto numNodesWithDefaultMode = 0;
    for (auto const& node : getNewNodes(createGenomeWithNewNodes())) {
        if (auto hasDefaultMode = hasDefaultCellTypeMode(node)) {
            ++numNodesWithMode;
            if (*hasDefaultMode) {
                ++numNodesWithDefaultMode;
            }
        }
    }
    ASSERT_LT(0, numNodesWithMode);

    // Averaged over the cell types with a mode, a uniformly drawn mode is the default one in 3 of 8 cases.
    auto defaultModeFraction = static_cast<float>(numNodesWithDefaultMode) / static_cast<float>(numNodesWithMode);
    EXPECT_LT(0.2f, defaultModeFraction);
    EXPECT_GT(0.55f, defaultModeFraction);
}

TEST_P(NewNodeMutationTests, newNode_randomCellTypePropertiesOnlyWithRandomNeuralNetwork)
{
    auto numRandomNodes = 0;
    auto numRandomNodesWithChangedProperties = 0;
    for (auto const& node : getNewNodes(createGenomeWithNewNodes())) {
        auto defaultCellType = node._cellType;
        resetCellTypeProperties(defaultCellType);
        if (hasDefaultNeuralNetwork(node)) {
            EXPECT_TRUE(node._cellType == defaultCellType);
        } else {
            ++numRandomNodes;
            if (node._cellType != defaultCellType) {
                ++numRandomNodesWithChangedProperties;
            }
        }
    }

    // Some cell types and modes have no properties.
    EXPECT_LT(numRandomNodes / 2, numRandomNodesWithChangedProperties);
}

TEST_P(NewNodeMutationTests, newNode_passesValidation)
{
    auto genome = createGenomeWithNewNodes();

    auto validatedGenome = genome;
    DescValidationService::get().validateAndCorrect(validatedGenome);
    EXPECT_TRUE(genome == validatedGenome);
}
