#include <type_traits>
#include <variant>

#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class AddNodeMutationTests : public MutationTestsBase
{};

TEST_F(AddNodeMutationTests, addNodeMutation_addsExactlyOneNodePerPass)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().cellType(BaseGenomeDesc()), NodeDesc().cellType(BaseGenomeDesc())})});
    genome._mutationRates._addNodeMutation = AddNodeMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._genes.at(0)._nodes.size(), 3u);
}

TEST_F(AddNodeMutationTests, addNodeMutation_increasesNodeCount)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().cellType(BaseGenomeDesc())})});
    genome._mutationRates._addNodeMutation = AddNodeMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_GT(actualGenome._genes.at(0)._nodes.size(), genome._genes.at(0)._nodes.size());
}

TEST_F(AddNodeMutationTests, addNodeMutation_newNodesHaveDefaultAttributes)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().cellType(BaseGenomeDesc())})});
    genome._mutationRates._addNodeMutation = AddNodeMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 20; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    for (auto const& node : actualGenome._genes.at(0)._nodes) {
        // Added nodes get a random non-void cell type and random mode, but node-level attributes stay at their defaults.
        EXPECT_NE(node.getCellType(), CellType_Void);
        EXPECT_EQ(node._referenceAngle, 0.0f);
        EXPECT_EQ(node._color, 0);
        EXPECT_FALSE(node._constructor.has_value());
    }
}

TEST_F(AddNodeMutationTests, addNodeMutation_zeroProbabilityNoChange)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().cellType(BaseGenomeDesc()), NodeDesc().cellType(BaseGenomeDesc())})});
    genome._mutationRates._addNodeMutation = AddNodeMutationDesc().geneProbability(0.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome, genome);
}
