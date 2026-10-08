#include <random>
#include <ranges>

#include <gtest/gtest.h>

#include <Data/DescValidationService.h>
#include <Data/Descs.h>

#include <EngineInterface/SimulationFacade.h>

#include <EngineTestData/DescTestDataFactory.h>

#include "MutationTestsBase.h"

class GeneGraphProcessorParityTests : public MutationTestsBase
{};

TEST_F(GeneGraphProcessorParityTests, cpuCorrectionsMatchGpuCorrections)
{
    std::mt19937 randomEngine(1);
    for (auto run : std::views::iota(0, 200)) {
        auto genome = DescTestDataFactory::get().createRandomGenome(randomEngine);
        auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

        _simulationFacade->setSimulationData(data);
        _simulationFacade->testOnly_voidUnreachableNodes(1);
        _simulationFacade->testOnly_removeGeneCycles(1);
        _simulationFacade->testOnly_removeUnusedGenes(1);
        _simulationFacade->testOnly_limitGenesWithSeparation(1);

        auto expectedGenome = genome;
        DescValidationService::get().fixGenomeIssues(expectedGenome);
        EXPECT_EQ(expectedGenome._genes, getMutatedGenome()._genes) << "run " << run;
    }
}
