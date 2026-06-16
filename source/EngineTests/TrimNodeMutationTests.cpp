#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class TrimNodeMutationTests : public MutationTestsBase
{};

TEST_F(TrimNodeMutationTests, trimNodeMutation_removesExactlyOneNodePerPass)
{
    auto genome = GenomeDesc().genes(
        {GeneDesc().nodes({NodeDesc().cellType(BaseGenomeDesc()), NodeDesc().cellType(BaseGenomeDesc()), NodeDesc().cellType(BaseGenomeDesc())})});
    genome._mutationRates._trimNodeMutation = TrimNodeMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._genes.at(0)._nodes.size(), 2u);
}

TEST_F(TrimNodeMutationTests, trimNodeMutation_keepsAtLeastOneNode)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().cellType(BaseGenomeDesc()), NodeDesc().cellType(BaseGenomeDesc())})});
    genome._mutationRates._trimNodeMutation = TrimNodeMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome._genes.at(0)._nodes.size(), 1u);
}

TEST_F(TrimNodeMutationTests, trimNodeMutation_keepsFirstAndLastNonVoid)
{
    // Trimming a boundary node can expose an interior void node; it must be corrected to a non-void cell type.
    auto genome = GenomeDesc().genes({GeneDesc().nodes({
        NodeDesc().cellType(BaseGenomeDesc()),
        NodeDesc().cellType(VoidGenomeDesc()),
        NodeDesc().cellType(BaseGenomeDesc()),
    })});
    genome._mutationRates._trimNodeMutation = TrimNodeMutationDesc().geneProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    auto const& nodes = actualGenome._genes.at(0)._nodes;
    ASSERT_GE(nodes.size(), 1u);
    EXPECT_NE(nodes.front().getCellType(), CellType_Void);
    EXPECT_NE(nodes.back().getCellType(), CellType_Void);
}

TEST_F(TrimNodeMutationTests, trimNodeMutation_zeroProbabilityNoChange)
{
    auto genome = GenomeDesc().genes(
        {GeneDesc().nodes({NodeDesc().cellType(BaseGenomeDesc()), NodeDesc().cellType(BaseGenomeDesc()), NodeDesc().cellType(BaseGenomeDesc())})});
    genome._mutationRates._trimNodeMutation = TrimNodeMutationDesc().geneProbability(0.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome, genome);
}
