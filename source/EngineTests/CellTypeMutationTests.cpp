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
    bool compareAllExceptCellType(GenomeDesc expected, GenomeDesc actual)
    {
        auto reset = [](GenomeDesc& genome) {
            genome._lineageId = 0;
            genome._prevLineageId = std::nullopt;
            genome._accumulatedMutations = 0.0f;
            for (auto& gene : genome._genes) {
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
    genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().eventProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    auto const& cellType = actualGenome._genes.at(0)._nodes.at(0)._cellType;

    EXPECT_FALSE(std::holds_alternative<SensorGenomeDesc>(cellType));

    // The new cell type must be initialized with default attribute values.
    std::visit([](auto const& cellTypeData) { EXPECT_EQ(cellTypeData, std::decay_t<decltype(cellTypeData)>{}); }, cellType);
}

TEST_F(CellTypeMutationTests, cellTypeMutation_doesNotChangeExceptCellType)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().eventProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    EXPECT_TRUE(compareAllExceptCellType(genome, actualGenome));
}
