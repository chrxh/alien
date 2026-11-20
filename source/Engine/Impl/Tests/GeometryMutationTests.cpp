#include <algorithm>
#include <ranges>
#include <vector>

#include <gtest/gtest.h>

#include <Data/CellTypeConstants.h>
#include <Data/Descs.h>

#include <Engine/Interface/SimulationFacade.h>

#include "MutationTestsBase.h"

class GeometryMutationTests : public MutationTestsBase
{
};

TEST_F(GeometryMutationTests, geometryMutation_changesShapeStiffness)
{
    auto genome = createTestGenome();
    for (auto& gene : genome._genes) {
        gene._stiffness = 0.5f;
    }
    genome._mutationRates._geometryMutations[0] = GeometryMutationDesc().geneProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();

    ASSERT_EQ(genome._genes.size(), actualGenome._genes.size());
    ASSERT_FALSE(genome._genes.empty());

    auto const& expectedGene = genome._genes.front();
    auto const& actualGene = actualGenome._genes.front();
    EXPECT_NE(expectedGene._shape, actualGene._shape);
    EXPECT_NE(expectedGene._stiffness, actualGene._stiffness);

    for (auto const& gene : actualGenome._genes) {
        EXPECT_GE(gene._stiffness, Const::GeneStiffness_Min);
        EXPECT_LE(gene._stiffness, Const::GeneStiffness_Max);
        EXPECT_GE(gene._shape, 0);
        EXPECT_LT(gene._shape, ConstructorShape_Count);
    }
}

TEST_F(GeometryMutationTests, geometryMutation_zeroProbabilityNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._geometryMutations[0] = GeometryMutationDesc().geneProbability(0.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);
    genome._mutationRates._geometryMutations[1] = GeometryMutationDesc().geneProbability(0.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    ASSERT_EQ(genome._genes.size(), actualGenome._genes.size());
    for (auto const& [expectedGene, actualGene] : std::views::zip(genome._genes, actualGenome._genes)) {
        EXPECT_EQ(expectedGene._shape, actualGene._shape);
        EXPECT_EQ(expectedGene._stiffness, actualGene._stiffness);
    }
}

TEST_F(GeometryMutationTests, geometryMutation_changesNodesOnlyAtRestartConstruction)
{
    auto genome = createTestGenome();
    genome._mutationRates._geometryMutations[0] = GeometryMutationDesc().geneProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);
    genome._mutationRates._geometryMutations[1] = GeometryMutationDesc().geneProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    ASSERT_EQ(genome._genes.size(), actualGenome._genes.size());
    for (auto const& [expectedGene, actualGene] : std::views::zip(genome._genes, actualGenome._genes)) {
        ASSERT_EQ(expectedGene._nodes.size(), actualGene._nodes.size());
        for (auto const& [expectedNode, actualNode] : std::views::zip(expectedGene._nodes, actualGene._nodes)) {
            auto actualNodeWithExpectedRestart = actualNode;
            actualNodeWithExpectedRestart._restartConstruction = expectedNode._restartConstruction;
            EXPECT_EQ(expectedNode, actualNodeWithExpectedRestart);
        }
        EXPECT_EQ(expectedGene._name, actualGene._name);
        EXPECT_EQ(expectedGene._homogeneousCellType, actualGene._homogeneousCellType);
    }
}

TEST_F(GeometryMutationTests, geometryMutation_switchesRestartConstructionAtOneNode)
{
    auto const NumGenes = 50;
    auto const NumNodes = 5;

    auto genome = GenomeDesc().genes(std::vector<GeneDesc>(NumGenes, GeneDesc().nodes(std::vector<NodeDesc>(NumNodes))));
    genome._mutationRates._geometryMutations[0] = GeometryMutationDesc().geneProbability(1.0f).enumChangeProbability(1.0f);

    _simulationFacade->setSimulationData(ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome));
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(NumGenes, actualGenome._genes.size());
    std::vector<int> numSwitchesByNodeIndex(NumNodes, 0);
    for (auto const& gene : actualGenome._genes) {
        ASSERT_EQ(NumNodes, gene._nodes.size());
        EXPECT_EQ(1, std::ranges::count_if(gene._nodes, &NodeDesc::_restartConstruction));
        for (auto const& [index, node] : gene._nodes | std::views::enumerate) {
            if (node._restartConstruction) {
                ++numSwitchesByNodeIndex.at(index);
            }
        }
    }
    EXPECT_EQ(0, numSwitchesByNodeIndex.front());
    for (auto numSwitches : numSwitchesByNodeIndex | std::views::drop(1)) {
        EXPECT_GT(numSwitches, 0);
    }
}
