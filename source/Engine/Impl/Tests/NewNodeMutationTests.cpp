#include <gtest/gtest.h>

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
