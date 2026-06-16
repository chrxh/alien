#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class AddNodeMutationTests : public MutationTestsBase
{};

TEST_F(AddNodeMutationTests, addNodeMutation_addsExactlyOneNodePerPass)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    genome._mutationRates._addNodeMutation = AddNodeMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    auto const& nodes = actualGenome._genes.at(0)._nodes;
    EXPECT_EQ(3, nodes.size());
    for (auto const& node : nodes) {
        // Each node has a non-void cell type; the added node keeps default node-level attributes.
        EXPECT_NE(CellType_Void, node.getCellType());
        EXPECT_EQ(0.0f, node._referenceAngle);
        EXPECT_FALSE(node._constructor.has_value());
    }
}

TEST_F(AddNodeMutationTests, addNodeMutation_zeroProbabilityNoChange)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    genome._mutationRates._addNodeMutation = AddNodeMutationDesc().geneProbability(0.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}
