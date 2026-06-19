#include <optional>
#include <type_traits>
#include <variant>

#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class CellTypeModeMutationTests : public MutationTestsBase
{
protected:
    // Resets the active cell type mode (selected mode and its data) to a canonical value while preserving all other attributes.
    static void resetCellTypeMode(CellTypeGenomeDesc& cellType)
    {
        std::visit(
            [](auto& cellTypeData) {
                if constexpr (requires { cellTypeData._mode; }) {
                    cellTypeData._mode = std::decay_t<decltype(cellTypeData._mode)>{};
                }
            },
            cellType);
    }

    bool compareAllExceptCellTypeMode(GenomeDesc expected, GenomeDesc actual)
    {
        auto reset = [](GenomeDesc& genome) {
            for (auto& gene : genome._genes) {
                for (auto& node : gene._nodes) {
                    resetCellTypeMode(node._cellType);
                }
            }
        };
        reset(expected);
        reset(actual);
        return expected == actual;
    }
};

TEST_F(CellTypeModeMutationTests, cellTypeModeMutation_changesModeToDefaults)
{
    auto genome = GenomeDesc().genes(
        {GeneDesc().nodes({NodeDesc().cellType(MuscleGenomeDesc().mode(AutoBendingGenomeDesc().maxAngleDeviation(0.55f).forwardBackwardRatio(0.25f)))})});
    genome._mutationRates._cellTypeModeMutation = CellTypeModeMutationDesc().nodeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    auto const& muscle = std::get<MuscleGenomeDesc>(actualGenome._genes.at(0)._nodes.at(0)._cellType);

    EXPECT_NE(muscle.getMode(), MuscleMode_AutoBending);

    // The new mode must be initialized with default attribute values.
    std::visit([](auto const& modeData) { EXPECT_EQ(modeData, std::decay_t<decltype(modeData)>{}); }, muscle._mode);
}

TEST_F(CellTypeModeMutationTests, cellTypeModeMutation_doesNotChangeExceptMode)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypeModeMutation = CellTypeModeMutationDesc().nodeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    EXPECT_TRUE(compareAllExceptCellTypeMode(genome, actualGenome));
}
