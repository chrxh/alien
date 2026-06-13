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
    // The first and last node of a gene cannot be void, so the toggling is observed on interior nodes:
    // a non-void node becomes void and a void node becomes non-void with default attribute values.
    auto genome = GenomeDesc().genes({GeneDesc().nodes({
        NodeDesc().cellType(BaseGenomeDesc()),
        NodeDesc().cellType(MuscleGenomeDesc()),
        NodeDesc().cellType(VoidGenomeDesc()),
        NodeDesc().cellType(BaseGenomeDesc()),
    })});
    genome._mutationRates._voidMutation = VoidMutationDesc().eventProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    auto const& nodes = actualGenome._genes.at(0)._nodes;

    EXPECT_EQ(nodes.at(1).getCellType(), CellType_Void);

    auto const& revived = nodes.at(2)._cellType;
    EXPECT_FALSE(std::holds_alternative<VoidGenomeDesc>(revived));
    std::visit([](auto const& cellTypeData) { EXPECT_EQ(cellTypeData, std::decay_t<decltype(cellTypeData)>{}); }, revived);
}

TEST_F(VoidMutationTests, voidMutation_keepsBoundaryNodesNonVoid)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({
        NodeDesc().cellType(BaseGenomeDesc()),
        NodeDesc().cellType(MuscleGenomeDesc()),
        NodeDesc().cellType(MuscleGenomeDesc()),
        NodeDesc().cellType(BaseGenomeDesc()),
    })});
    genome._mutationRates._voidMutation = VoidMutationDesc().eventProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    auto const& nodes = actualGenome._genes.at(0)._nodes;

    EXPECT_NE(nodes.front().getCellType(), CellType_Void);
    EXPECT_NE(nodes.back().getCellType(), CellType_Void);
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

    // With zero probability no node changes its cell type.
    EXPECT_EQ(actualGenome._genes.at(0)._nodes.at(1).getCellType(), CellType_Muscle);
}
