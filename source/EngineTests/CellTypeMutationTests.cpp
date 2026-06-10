#include <type_traits>
#include <variant>

#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class CellTypeMutationTests : public MutationTestsBase
{};

TEST_F(CellTypeMutationTests, cellTypeMutation_changesCellTypeToDefaults)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().cellType(SensorGenomeDesc().autoTrigger(false).minRange(123).maxRange(200))})});
    genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().eventProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    auto const& cellType = actualGenome._genes.at(0)._nodes.at(0)._cellType;

    // The cell type must have changed away from the original Sensor type.
    EXPECT_FALSE(std::holds_alternative<SensorGenomeDesc>(cellType));

    // The new cell type must be initialized with default attribute values.
    std::visit([](auto const& cellTypeData) { EXPECT_EQ(cellTypeData, std::decay_t<decltype(cellTypeData)>{}); }, cellType);
}

TEST_F(CellTypeMutationTests, cellTypeMutation_zeroProbabilityKeepsCellType)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().cellType(DetonatorGenomeDesc().countdown(42))})});
    genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().eventProbability(0.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    auto const& cellType = actualGenome._genes.at(0)._nodes.at(0)._cellType;

    ASSERT_TRUE(std::holds_alternative<DetonatorGenomeDesc>(cellType));
    EXPECT_EQ(std::get<DetonatorGenomeDesc>(cellType)._countdown, 42);
}

TEST_F(CellTypeMutationTests, cellTypeMutation_resultingCellTypesAlwaysHaveDefaults)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().cellType(BaseGenomeDesc())})});
    genome._mutationRates._cellTypeMutation = CellTypeMutationDesc().eventProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);

    // After every mutation the freshly assigned cell type must carry default attribute values, regardless of which type was chosen.
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
        auto actualGenome = getMutatedGenome();
        auto const& cellType = actualGenome._genes.at(0)._nodes.at(0)._cellType;
        std::visit([](auto const& cellTypeData) { EXPECT_EQ(cellTypeData, std::decay_t<decltype(cellTypeData)>{}); }, cellType);
    }
}
