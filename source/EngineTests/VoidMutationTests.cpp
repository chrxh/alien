#include <type_traits>
#include <variant>

#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class VoidMutationTests : public MutationTestsBase
{};

TEST_F(VoidMutationTests, voidMutation_nonVoidBecomesVoid)
{
    // The first and last node of a gene cannot be void, so only the interior node is toggled.
    auto genome = GenomeDesc().genes({GeneDesc().nodes({
        NodeDesc().cellType(BaseGenomeDesc()),
        NodeDesc().cellType(MuscleGenomeDesc()),
        NodeDesc().cellType(BaseGenomeDesc()),
    })});
    genome._mutationRates._voidMutation = VoidMutationDesc().eventProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    auto const& nodes = actualGenome._genes.at(0)._nodes;

    EXPECT_EQ(nodes.at(0), genome._genes.at(0)._nodes.at(0));
    EXPECT_EQ(nodes.at(1).getCellType(), CellType_Void);
    EXPECT_EQ(nodes.at(2), genome._genes.at(0)._nodes.at(2));
}

TEST_F(VoidMutationTests, voidMutation_voidBecomesNonVoid)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({
        NodeDesc().cellType(BaseGenomeDesc()),
        NodeDesc().cellType(VoidGenomeDesc()),
        NodeDesc().cellType(BaseGenomeDesc()),
    })});
    genome._mutationRates._voidMutation = VoidMutationDesc().eventProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    auto const& nodes = actualGenome._genes.at(0)._nodes;

    EXPECT_EQ(nodes.at(0), genome._genes.at(0)._nodes.at(0));
    EXPECT_EQ(nodes.at(2), genome._genes.at(0)._nodes.at(2));

    // The void node is turned into a non-void cell type with default attribute values.
    auto const& revived = nodes.at(1)._cellType;
    EXPECT_FALSE(std::holds_alternative<VoidGenomeDesc>(revived));
    std::visit([](auto const& cellTypeData) { EXPECT_EQ(cellTypeData, std::decay_t<decltype(cellTypeData)>{}); }, revived);
}

TEST_F(VoidMutationTests, voidMutation_zeroProbabilityNoChange)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({
        NodeDesc().cellType(BaseGenomeDesc()),
        NodeDesc().cellType(MuscleGenomeDesc()),
        NodeDesc().cellType(BaseGenomeDesc()),
    })});
    genome._mutationRates._voidMutation = VoidMutationDesc().eventProbability(0.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome, genome);
}
