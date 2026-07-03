#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class AddNodeMutationTests : public MutationTestsBase
{};

TEST_F(AddNodeMutationTests, addNodeMutation_insertsBeforeEveryNode)
{
    // With node probability 1, each of the numNodes + 1 insertion slots [0, numNodes] gets a new node (position numNodes
    // appends at the end), so a 2-node gene grows to 2 + 3 = 5 nodes in one pass.
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    genome._mutationRates._addNodeMutation = AddNodeMutationDesc().nodeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    auto const& nodes = actualGenome._genes.at(0)._nodes;
    EXPECT_EQ(5, nodes.size());
    for (auto const& node : nodes) {
        // Each node has a non-void cell type; the added nodes keep default node-level attributes.
        EXPECT_NE(CellType_Void, node.getCellType());
        EXPECT_EQ(0.0f, node._referenceAngle);
        EXPECT_FALSE(node._constructor.has_value());
    }
}

TEST_F(AddNodeMutationTests, addNodeMutation_zeroProbabilityNoChange)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    genome._mutationRates._addNodeMutation = AddNodeMutationDesc().nodeProbability(0.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}
