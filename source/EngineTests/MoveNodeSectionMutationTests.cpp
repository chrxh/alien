#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Descs.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class MoveNodeSectionMutationTests : public MutationTestsBase
{
protected:
    int countNodes(GenomeDesc const& genome) const
    {
        int result = 0;
        for (auto const& gene : genome._genes) {
            result += toInt(gene._nodes.size());
        }
        return result;
    }
};

TEST_F(MoveNodeSectionMutationTests, moveNodeSectionMutation_zeroProbabilityNoChange)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });
    genome._mutationRates._moveNodeSectionMutation = MoveNodeSectionMutationDesc().geneProbability(0.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}

TEST_F(MoveNodeSectionMutationTests, moveNodeSectionMutation_singleGeneConservesNodeCount)
{
    // A move never adds or removes nodes overall; in a single gene the section can only be relocated within the same gene, so the
    // node count is preserved and at least one node is always left behind.
    int constexpr numNodes = 5;
    std::vector<NodeDesc> nodes(numNodes);
    auto genome = GenomeDesc().genes({GeneDesc().nodes(nodes)});
    genome._mutationRates._moveNodeSectionMutation = MoveNodeSectionMutationDesc().geneProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(1, actualGenome._genes.size());
    EXPECT_EQ(numNodes, countNodes(actualGenome));
}

TEST_F(MoveNodeSectionMutationTests, moveNodeSectionMutation_repeatedMutationConservesNodesAndGenes)
{
    // Repeated move-node-section mutations keep the total node count constant, never remove genes, never leave a gene empty and
    // never produce void boundary nodes.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc(), NodeDesc()}),
        GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc()}),
    });
    genome._mutationRates._moveNodeSectionMutation = MoveNodeSectionMutationDesc().geneProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 5; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(2, actualGenome._genes.size());
    EXPECT_EQ(countNodes(genome), countNodes(actualGenome));
    for (auto const& gene : actualGenome._genes) {
        ASSERT_FALSE(gene._nodes.empty());
        EXPECT_NE(CellType_Void, gene._nodes.front().getCellType());
        EXPECT_NE(CellType_Void, gene._nodes.back().getCellType());
    }
}
