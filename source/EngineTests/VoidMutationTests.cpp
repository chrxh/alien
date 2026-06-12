#include <type_traits>
#include <variant>

#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class VoidMutationTests : public MutationTestsBase
{
protected:
    // The first and last node of a gene cannot be void, so interior nodes are used to observe the toggling.
    GenomeDesc createBoundedGenome(CellTypeGenomeDesc interiorCellType) const
    {
        return GenomeDesc().genes({GeneDesc().nodes({
            NodeDesc().cellType(BaseGenomeDesc()),
            NodeDesc().cellType(interiorCellType),
            NodeDesc().cellType(BaseGenomeDesc()),
        })});
    }
};

TEST_F(VoidMutationTests, voidMutation_nonVoidBecomesVoid)
{
    auto genome = createBoundedGenome(MuscleGenomeDesc());
    genome._mutationRates._voidMutation = VoidMutationDesc().eventProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    auto const& nodes = actualGenome._genes.at(0)._nodes;

    // The interior node was turned void, the boundary nodes must stay non-void.
    EXPECT_NE(nodes.at(0).getCellType(), CellType_Void);
    EXPECT_EQ(nodes.at(1).getCellType(), CellType_Void);
    EXPECT_NE(nodes.at(2).getCellType(), CellType_Void);
}

TEST_F(VoidMutationTests, voidMutation_voidBecomesNonVoidWithDefaults)
{
    auto genome = createBoundedGenome(VoidGenomeDesc());
    genome._mutationRates._voidMutation = VoidMutationDesc().eventProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    auto const& cellType = actualGenome._genes.at(0)._nodes.at(1)._cellType;

    EXPECT_FALSE(std::holds_alternative<VoidGenomeDesc>(cellType));

    // The new cell type must be initialized with default attribute values.
    std::visit([](auto const& cellTypeData) { EXPECT_EQ(cellTypeData, std::decay_t<decltype(cellTypeData)>{}); }, cellType);
}

TEST_F(VoidMutationTests, voidMutation_keepsBoundaryNodesNonVoid)
{
    auto genome = createTestGenome();
    genome._mutationRates._voidMutation = VoidMutationDesc().eventProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    for (auto const& gene : actualGenome._genes) {
        ASSERT_FALSE(gene._nodes.empty());
        EXPECT_NE(gene._nodes.front().getCellType(), CellType_Void);
        EXPECT_NE(gene._nodes.back().getCellType(), CellType_Void);
    }
}

TEST_F(VoidMutationTests, voidMutation_zeroProbabilityNoChange)
{
    auto genome = createBoundedGenome(MuscleGenomeDesc());
    genome._mutationRates._voidMutation = VoidMutationDesc().eventProbability(0.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    auto const& nodes = actualGenome._genes.at(0)._nodes;

    // With zero probability no node changes its cell type.
    EXPECT_NE(nodes.at(0).getCellType(), CellType_Void);
    EXPECT_EQ(nodes.at(1).getCellType(), CellType_Muscle);
    EXPECT_NE(nodes.at(2).getCellType(), CellType_Void);
}
