#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class DeleteNodeMutationTests : public MutationTestsBase
{};

TEST_F(DeleteNodeMutationTests, deleteNodeMutation_removesExactlyOneNodePerPass)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc()})});
    genome._mutationRates._deleteNodeMutation = DeleteNodeMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(2, actualGenome._genes.at(0)._nodes.size());
}

TEST_F(DeleteNodeMutationTests, deleteNodeMutation_keepsAtLeastOneNode)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    genome._mutationRates._deleteNodeMutation = DeleteNodeMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 5; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(1, actualGenome._genes.at(0)._nodes.size());
}

TEST_F(DeleteNodeMutationTests, deleteNodeMutation_keepsFirstAndLastNonVoid)
{
    // Deleting a boundary node can expose the interior void node; it must be corrected to a non-void cell type.
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc().cellType(VoidGenomeDesc()), NodeDesc()})});
    genome._mutationRates._deleteNodeMutation = DeleteNodeMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 5; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    auto const& nodes = actualGenome._genes.at(0)._nodes;
    EXPECT_NE(CellType_Void, nodes.front().getCellType());
    EXPECT_NE(CellType_Void, nodes.back().getCellType());
}

TEST_F(DeleteNodeMutationTests, deleteNodeMutation_zeroProbabilityNoChange)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc()})});
    genome._mutationRates._deleteNodeMutation = DeleteNodeMutationDesc().geneProbability(0.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(genome, actualGenome);
}
