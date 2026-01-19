#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class GarbageCollectorTests : public IntegrationTestFramework
{
public:
    GarbageCollectorTests()
        : IntegrationTestFramework()
    {}

    ~GarbageCollectorTests() = default;
};

enum class CleanupAction
{
    CleanupAfterTimestep,
    CleanupAfterDataManipulation,
    ResizeArrays,
};

class GarbageCollectorTests_AllCleanupActions
    : public GarbageCollectorTests
    , public testing::WithParamInterface<CleanupAction>
{};

INSTANTIATE_TEST_SUITE_P(
    GarbageCollectorTests_AllCleanupActions,
    GarbageCollectorTests_AllCleanupActions,
    ::testing::Values(CleanupAction::CleanupAfterTimestep, CleanupAction::CleanupAfterDataManipulation, CleanupAction::ResizeArrays));

TEST_P(GarbageCollectorTests_AllCleanupActions, cleanupAfterTimestep_cellsAndParticles)
{
    auto cleanupAction = GetParam();

    auto& numberGen = NumberGenerator::get();

    auto data = DescriptionEditService::get().createHex(DescriptionEditService::CreateHexParameters().layers(10).center({100.0f, 100.0}));
    for (int i = 0; i < 100; ++i) {
        data._energies.emplace_back(EnergyDesc()
                                         .pos({numberGen.getRandomFloat(0.0f, 100.0f), numberGen.getRandomFloat(0.0f, 100.0f)})
                                         .vel({numberGen.getRandomFloat(-1.0f, 1.0f), numberGen.getRandomFloat(-1.0f, 1.0f)})
                                         .energy(numberGen.getRandomFloat(0.0f, 100.0f)));
    }
    _simulationFacade->setSimulationData(data);

    switch (cleanupAction) {
    case CleanupAction::CleanupAfterTimestep:
        _simulationFacade->testOnly_cleanupAfterTimestep();
        break;
    case CleanupAction::CleanupAfterDataManipulation:
        _simulationFacade->testOnly_cleanupAfterDataManipulation();
        break;
    case CleanupAction::ResizeArrays:
        _simulationFacade->testOnly_resizeArrays(ArraySizesForGpu{.objectArray = 1000, .energyArray = 1000, .heap = 1000000});
        break;
    }
    EXPECT_TRUE(_simulationFacade->testOnly_arePointersValid());
}

TEST_P(GarbageCollectorTests_AllCleanupActions, cleanupAfterTimestep_memoryCellsWithMemoryNodes)
{
    auto cleanupAction = GetParam();

    // Create a genome with memory cell type nodes that have memory entries
    auto genome = GenomeDesc().genes({GeneDesc().separation(true).nodes({
        NodeDesc().cellType(MemoryGenomeDesc().signalEntries({SignalEntryGenomeDesc()})),
        NodeDesc().cellType(
            MemoryGenomeDesc().signalEntries({SignalEntryGenomeDesc(), SignalEntryGenomeDesc(), SignalEntryGenomeDesc()})),
    })});

    auto data = Desc().addCreature(
        {
            ObjectDesc().pos({100.0f, 100.0f}).type(CellDesc().cellType(MemoryDesc().signalEntries({SignalEntryDesc()}))),
            ObjectDesc().pos({101.0f, 100.0f}).type(CellDesc().cellType(MemoryDesc().signalEntries({SignalEntryDesc(), SignalEntryDesc()}))),
        },
        CreatureDesc(),
        genome);
    _simulationFacade->setSimulationData(data);

    switch (cleanupAction) {
    case CleanupAction::CleanupAfterTimestep:
        _simulationFacade->testOnly_cleanupAfterTimestep();
        break;
    case CleanupAction::CleanupAfterDataManipulation:
        _simulationFacade->testOnly_cleanupAfterDataManipulation();
        break;
    case CleanupAction::ResizeArrays:
        _simulationFacade->testOnly_resizeArrays(ArraySizesForGpu{.objectArray = 1000, .energyArray = 1000, .heap = 1000000});
        break;
    }
    EXPECT_TRUE(_simulationFacade->testOnly_arePointersValid());
}
