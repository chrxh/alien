#include <optional>

#include <gtest/gtest.h>

#include <Data/CellTypeConstants.h>
#include <Data/DescValidationService.h>
#include <Data/Descs.h>

#include <Engine/Interface/SimulationFacade.h>

#include "MutationTestsBase.h"

namespace
{
    auto constexpr NumGenes = 400;
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

    std::vector<NodeDesc> getNewNodes(GenomeDesc const& genome)
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
    std::optional<bool> hasDefaultCellTypeMode(NodeDesc const& node)
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
    // A probability of 1 still fails when the random number is exactly 1.
    ASSERT_LE(NumGenes - 10, toInt(newNodes.size()));

    auto numDefaultNodes = std::ranges::count_if(newNodes, hasDefaultNeuralNetwork);
    auto defaultFraction = static_cast<float>(numDefaultNodes) / static_cast<float>(newNodes.size());
    EXPECT_LT(0.35f, defaultFraction);
    EXPECT_GT(0.65f, defaultFraction);
}

TEST_P(NewNodeMutationTests, newNode_randomNeuralNetworkIsSparse)
{
    auto numRandomNodes = 0;
    auto numWeights = 0;
    auto numBiases = 0;
    auto numActivationFunctions = 0;
    for (auto const& node : getNewNodes(createGenomeWithNewNodes())) {
        if (hasDefaultNeuralNetwork(node)) {
            continue;
        }
        auto const& neuralNetwork = node._neuralNetwork;
        ++numRandomNodes;
        numWeights += toInt(std::ranges::count_if(neuralNetwork._weights, [](auto const& weight) { return weight.getValue() != 0.0f; }));
        numBiases += toInt(std::ranges::count_if(neuralNetwork._biases, [](auto const& bias) { return bias != 0.0f; }));
        numActivationFunctions +=
            toInt(std::ranges::count_if(neuralNetwork._activationFunctions, [](auto const& function) { return function != ActivationFunction_Identity; }));
    }

    // Random weights, biases and activation functions occur, but fewer than 1 in 8 values differs from 0 or Identity.
    auto expectSparse = [&](int numRandomValues, int numValuesPerNode) {
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

TEST_P(NewNodeMutationTests, newNode_creatureFilterAdmitsSomeSize)
{
    auto admitsSomeSize = [](auto const& filter) {
        auto maxNumCells = filter._maxNumCells.value_or(0);
        return maxNumCells == 0 || filter._minNumCells.value_or(0) <= maxNumCells;
    };
    for (auto const& node : getNewNodes(createGenomeWithNewNodes())) {
        if (auto sensor = std::get_if<SensorGenomeDesc>(&node._cellType)) {
            if (auto detectCreature = std::get_if<DetectCreatureGenomeDesc>(&sensor->_mode)) {
                EXPECT_TRUE(admitsSomeSize(*detectCreature));
            }
        }
        if (auto reconnector = std::get_if<ReconnectorGenomeDesc>(&node._cellType)) {
            if (auto reconnectCreature = std::get_if<ReconnectCreatureGenomeDesc>(&reconnector->_mode)) {
                EXPECT_TRUE(admitsSomeSize(*reconnectCreature));
            }
        }
    }
}

TEST_P(NewNodeMutationTests, newNode_passesValidation)
{
    auto genome = createGenomeWithNewNodes();

    auto validatedGenome = genome;
    DescValidationService::get().validateAndCorrect(validatedGenome);
    EXPECT_TRUE(genome == validatedGenome);
}

class NewNodeConnectionWeightTests : public MutationTestsBase
{
protected:
    static auto constexpr PredecessorConnectionIndex = 1;
    static auto constexpr BranchConnectionIndex = 2;

    // Every gene consists of one old node. Gene 0 additionally holds one constructor for each further gene if a constructor is given.
    GenomeDesc createGenomeWithExtendedGenes(std::optional<ConstructorGenomeDesc> const& constructor)
    {
        std::vector<GeneDesc> genes(NumGenes, GeneDesc().nodes({NodeDesc().referenceAngle(OldNodeAngle)}));
        if (constructor.has_value()) {
            std::vector<NodeDesc> constructorNodes;
            for (int geneIndex = 1; geneIndex < NumGenes; ++geneIndex) {
                constructorNodes.emplace_back(NodeDesc().referenceAngle(OldNodeAngle).constructor(ConstructorGenomeDesc(*constructor).geneIndex(geneIndex)));
            }
            genes.front() = GeneDesc().nodes(constructorNodes);
        }
        auto genome = GenomeDesc().genes(genes);
        genome._mutationRates._extendGeneMutation = ExtendGeneMutationDesc().geneProbability(1.0f);
        return mutate(genome);
    }

    GenomeDesc mutate(GenomeDesc const& genome)
    {
        _simulationFacade->setSimulationData(ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome));
        _simulationFacade->testOnly_mutate(1);
        return getMutatedGenome();
    }

    // Extend gene places the new node either before or after the old node.
    bool hasNewLastNode(GeneDesc const& gene) { return gene._nodes.front()._referenceAngle == OldNodeAngle; }

    NodeDesc const& getConstructorNode(GenomeDesc const& genome, int geneIndex)
    {
        auto const& nodes = genome._genes.front()._nodes;
        return *std::ranges::find_if(nodes, [&](auto const& node) { return node._constructor.has_value() && node._constructor->_geneIndex == geneIndex; });
    }

    bool hasWeight(NodeDesc const& node, int connectionIndex) { return node._neuralNetwork._connectionWeights.at(connectionIndex) != 0.0f; }

    void expectWeightsForHalfOfCandidates(int numWithWeights, int numCandidates)
    {
        ASSERT_LT(50, numCandidates);
        auto fraction = static_cast<float>(numWithWeights) / static_cast<float>(numCandidates);
        EXPECT_LT(0.3f, fraction);
        EXPECT_GT(0.7f, fraction);
    }
};

TEST_F(NewNodeConnectionWeightTests, newNodeInFront_successorGetsWeightInHalfOfTheCases)
{
    auto genome = createGenomeWithExtendedGenes(std::nullopt);

    auto numCandidates = 0;
    auto numWithWeights = 0;
    for (auto const& gene : genome._genes) {
        auto const& oldNode = hasNewLastNode(gene) ? gene._nodes.front() : gene._nodes.back();
        if (hasNewLastNode(gene)) {
            EXPECT_FALSE(hasWeight(oldNode, PredecessorConnectionIndex));
        } else {
            ++numCandidates;
            numWithWeights += hasWeight(oldNode, PredecessorConnectionIndex) ? 1 : 0;
        }
    }
    expectWeightsForHalfOfCandidates(numWithWeights, numCandidates);
}

TEST_F(NewNodeConnectionWeightTests, newLastNode_constructorGetsWeightsInHalfOfTheCases)
{
    auto genome = createGenomeWithExtendedGenes(ConstructorGenomeDesc());

    auto numCandidates = 0;
    auto numWithWeights = 0;
    for (int geneIndex = 1; geneIndex < NumGenes; ++geneIndex) {
        auto const& constructorNode = getConstructorNode(genome, geneIndex);
        auto const& gene = genome._genes.at(geneIndex);
        if (hasNewLastNode(gene)) {
            ++numCandidates;
            numWithWeights += hasWeight(constructorNode, BranchConnectionIndex) ? 1 : 0;
            EXPECT_FALSE(hasWeight(gene._nodes.front(), PredecessorConnectionIndex));
        } else {
            EXPECT_FALSE(hasWeight(constructorNode, BranchConnectionIndex));
        }
    }
    expectWeightsForHalfOfCandidates(numWithWeights, numCandidates);
}

TEST_F(NewNodeConnectionWeightTests, newLastNode_constructorWithSeparationGetsNoWeights)
{
    auto genome = createGenomeWithExtendedGenes(ConstructorGenomeDesc().separation(true));

    for (int geneIndex = 1; geneIndex < NumGenes; ++geneIndex) {
        EXPECT_FALSE(hasWeight(getConstructorNode(genome, geneIndex), BranchConnectionIndex));
    }
}

TEST_F(NewNodeConnectionWeightTests, newLastNode_firstNodeGetsWeightWithConcatenations)
{
    auto genome = createGenomeWithExtendedGenes(ConstructorGenomeDesc().separation(true).numConcatenations(2));

    auto numCandidates = 0;
    auto numWithWeights = 0;
    for (int geneIndex = 1; geneIndex < NumGenes; ++geneIndex) {
        auto const& gene = genome._genes.at(geneIndex);
        if (hasNewLastNode(gene)) {
            ++numCandidates;
            numWithWeights += hasWeight(gene._nodes.front(), PredecessorConnectionIndex) ? 1 : 0;
        }
    }
    expectWeightsForHalfOfCandidates(numWithWeights, numCandidates);
}

TEST_F(NewNodeConnectionWeightTests, addedNodes_successorAndConstructorGetWeightsInHalfOfTheCases)
{
    // The genes form a chain: the old node of each gene builds the next gene.
    std::vector<GeneDesc> genes;
    for (int geneIndex = 0; geneIndex < NumGenes; ++geneIndex) {
        auto node = NodeDesc().referenceAngle(OldNodeAngle);
        if (geneIndex + 1 < NumGenes) {
            node.constructor(ConstructorGenomeDesc().geneIndex(geneIndex + 1));
        }
        genes.emplace_back(GeneDesc().nodes({node}));
    }
    auto genome = GenomeDesc().genes(genes);
    genome._mutationRates._addNodeMutation = AddNodeMutationDesc().nodeProbability(1.0f);
    auto mutatedGenome = mutate(genome);

    // Add node inserts a new node before and after the old node of each gene. As a constructor, the old node gets weights at both
    // connections to the next gene, so only the old nodes without these weights show whether they get a weight as a successor.
    auto numCandidates = 0;
    auto numConstructorsWithWeights = 0;
    auto numConstructorsWithoutWeights = 0;
    auto numSuccessorsWithWeight = 0;
    for (auto const& gene : mutatedGenome._genes | std::views::take(NumGenes - 1)) {
        auto const& oldNode = gene._nodes.at(1);
        ++numCandidates;
        if (hasWeight(oldNode, BranchConnectionIndex)) {
            ++numConstructorsWithWeights;
        } else {
            ++numConstructorsWithoutWeights;
            numSuccessorsWithWeight += hasWeight(oldNode, PredecessorConnectionIndex) ? 1 : 0;
        }
    }
    expectWeightsForHalfOfCandidates(numConstructorsWithWeights, numCandidates);
    expectWeightsForHalfOfCandidates(numSuccessorsWithWeight, numConstructorsWithoutWeights);
}

TEST_F(NewNodeConnectionWeightTests, newGene_constructorGetsWeightsInHalfOfTheCases)
{
    auto genome = GenomeDesc().genes(std::vector<GeneDesc>(NumGenes, GeneDesc().nodes({NodeDesc().referenceAngle(OldNodeAngle)})));
    genome._mutationRates._addGeneMutation = AddGeneMutationDesc().geneProbability(1.0f);
    auto mutatedGenome = mutate(genome);

    auto numCandidates = 0;
    auto numWithWeights = 0;
    for (auto const& gene : mutatedGenome._genes) {
        auto const& node = gene._nodes.front();
        if (node._constructor.has_value()) {
            ++numCandidates;
            numWithWeights += hasWeight(node, BranchConnectionIndex) ? 1 : 0;
        }
    }
    expectWeightsForHalfOfCandidates(numWithWeights, numCandidates);
}
