#include <type_traits>
#include <variant>

#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class CellTypeMutationTests : public MutationTestsBase
{
protected:
    bool compareAllExceptCellTypeAndHomogeneousFlag(GenomeDesc expected, GenomeDesc actual)
    {
        auto reset = [](GenomeDesc& genome) {
            genome._lineageId = 0;
            genome._prevLineageId = std::nullopt;
            genome._accumulatedMutations = 0.0f;
            for (auto& gene : genome._genes) {
                gene._homogeneousCellType = false;  // canonicalize: the cell type mutation may also flip this flag
                for (auto& node : gene._nodes) {
                    node._cellType = BaseGenomeDesc{};  // canonicalize so everything except the cell type is compared
                }
            }
        };
        reset(expected);
        reset(actual);
        return expected == actual;
    }
};

TEST_F(CellTypeMutationTests, cellTypeMutation_changesCellTypeToDefaults)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().cellType(SensorGenomeDesc().autoTrigger(false).minRange(123).maxRange(200))})});
    genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().nodeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    auto const& cellType = actualGenome._genes.at(0)._nodes.at(0)._cellType;

    EXPECT_FALSE(std::holds_alternative<SensorGenomeDesc>(cellType));

    // The new cell type must be initialized with default attribute values.
    std::visit([](auto const& cellTypeData) { EXPECT_EQ(cellTypeData, std::decay_t<decltype(cellTypeData)>{}); }, cellType);
}

TEST_F(CellTypeMutationTests, cellTypeMutation_doesNotChangeExceptCellTypeAndHomogeneousFlag)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().nodeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    EXPECT_TRUE(compareAllExceptCellTypeAndHomogeneousFlag(genome, actualGenome));
}

TEST_F(CellTypeMutationTests, cellTypeMutation_keepsVoidNodesVoid)
{
    // The first and last node of a gene cannot be void, so put the void node in the middle.
    auto genome = GenomeDesc().genes({GeneDesc().nodes({
        NodeDesc().cellType(BaseGenomeDesc()),
        NodeDesc().cellType(VoidGenomeDesc()),
        NodeDesc().cellType(BaseGenomeDesc()),
    })});
    genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().nodeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    auto const& nodes = actualGenome._genes.at(0)._nodes;
    EXPECT_NE(nodes.at(0).getCellType(), CellType_Void);
    EXPECT_EQ(nodes.at(1).getCellType(), CellType_Void);
    EXPECT_NE(nodes.at(2).getCellType(), CellType_Void);
}

TEST_F(CellTypeMutationTests, cellTypeMutation_keepsNonVoidNodesNonVoid)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().nodeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    // The cell type mutation must never turn a non-void node into a void node or vice versa.
    ASSERT_EQ(genome._genes.size(), actualGenome._genes.size());
    for (size_t geneIndex = 0; geneIndex < genome._genes.size(); ++geneIndex) {
        auto const& expectedNodes = genome._genes.at(geneIndex)._nodes;
        auto const& actualNodes = actualGenome._genes.at(geneIndex)._nodes;
        ASSERT_EQ(expectedNodes.size(), actualNodes.size());
        for (size_t nodeIndex = 0; nodeIndex < expectedNodes.size(); ++nodeIndex) {
            auto expectedIsVoid = expectedNodes.at(nodeIndex).getCellType() == CellType_Void;
            auto actualIsVoid = actualNodes.at(nodeIndex).getCellType() == CellType_Void;
            EXPECT_EQ(expectedIsVoid, actualIsVoid);
        }
    }
}

TEST_F(CellTypeMutationTests, cellTypeMutation_changesHomogeneousCellTypeFlag)
{
    auto genome = GenomeDesc().genes({GeneDesc().homogeneousCellType(false).nodes({NodeDesc().cellType(BaseGenomeDesc())})});
    genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().nodeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();

    EXPECT_TRUE(actualGenome._genes.at(0)._homogeneousCellType);
}
