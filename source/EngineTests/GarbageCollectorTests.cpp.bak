#include <gtest/gtest.h>

#include "EngineInterface/NumberGenerator.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Description.h"
#include "EngineInterface/SimulationFacade.h"
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

TEST_P(GarbageCollectorTests_AllCleanupActions, cleanupAfterTimestep)
{
    auto cleanupAction = GetParam();

    auto& numberGen = NumberGenerator::get();

    auto data = DescriptionEditService::get().createHex(DescriptionEditService::CreateHexParameters().layers(10).center({100.0f, 100.0}));
    for (int i = 0; i < 100; ++i) {
        data._particles.emplace_back(ParticleDescription()
                             .pos({numberGen.getRandomFloat(0.0f, 100.0f), numberGen.getRandomFloat(0.0f, 100.0f)})
                             .vel({numberGen.getRandomFloat(-1.0f, 1.0f), numberGen.getRandomFloat(-1.0f, 1.0f)})
                             .energy(numberGen.getRandomFloat(0.0f, 100.0f)));
    }
    _simulationFacade->setSimulationData(data);

    switch (cleanupAction){
    case CleanupAction::CleanupAfterTimestep:
        _simulationFacade->testOnly_cleanupAfterTimestep();
    case CleanupAction::CleanupAfterDataManipulation:
        _simulationFacade->testOnly_cleanupAfterDataManipulation();
    case CleanupAction::ResizeArrays:
        _simulationFacade->testOnly_resizeArrays(ArraySizesForGpu{.cellArray = 1000, .particleArray = 1000, .heap = 1000000});
    }
    EXPECT_TRUE(_simulationFacade->testOnly_areArraysValid());
}
