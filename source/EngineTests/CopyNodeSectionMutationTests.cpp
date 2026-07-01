#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class CopyNodeSectionMutationTests : public MutationTestsBase
{};

namespace
{
    int countNodes(GenomeDesc const& genome)
    {
        int result = 0;
        for (auto const& gene : genome._genes) {
            result += static_cast<int>(gene._nodes.size());
        }
        return result;
    }
}

TEST_F(CopyNodeSectionMutationTests, copyNodeSectionMutation_zeroProbabilityNoChange)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc()}),
        GeneDesc().nodes({NodeDesc()}),
    });
    genome._mutationRates._copyNodeSectionMutation = CopyNodeSectionMutationDesc().geneProbability(0.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}

TEST_F(CopyNodeSectionMutationTests, copyNodeSectionMutation_insertsSectionIntoSingleGene)
{
    // With a probability of 1 the single gene always copies a section (length in [1, numNodes]) into itself, so the node count
    // grows by at least one node and by at most the gene's length.
    int constexpr numNodes = 5;
    std::vector<NodeDesc> nodes(numNodes);
    auto genome = GenomeDesc().genes({GeneDesc().nodes(nodes)});
    genome._mutationRates._copyNodeSectionMutation = CopyNodeSectionMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    ASSERT_EQ(1, actualGenome._genes.size());
    auto actualNumNodes = countNodes(actualGenome);
    EXPECT_GT(actualNumNodes, numNodes);
    EXPECT_LE(actualNumNodes, 2 * numNodes);
}

TEST_F(CopyNodeSectionMutationTests, copyNodeSectionMutation_repeatedMutationGrowsGenome)
{
    // Repeated copy-node-section mutations keep growing the genome without ever removing genes or producing void boundary nodes.
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc()}),
        GeneDesc().nodes({NodeDesc(), NodeDesc()}),
    });
    genome._mutationRates._copyNodeSectionMutation = CopyNodeSectionMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 5; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(2, actualGenome._genes.size());
    EXPECT_GT(countNodes(actualGenome), countNodes(genome));
    for (auto const& gene : actualGenome._genes) {
        ASSERT_FALSE(gene._nodes.empty());
        EXPECT_NE(CellType_Void, gene._nodes.front().getCellType());
        EXPECT_NE(CellType_Void, gene._nodes.back().getCellType());
    }
}
