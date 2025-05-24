#include <gtest/gtest.h>

#include "Base/NumberGenerator.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GenomeDescriptionConverterService.h"
#include "EngineInterface/StatisticsRawData.h"

#include "IntegrationTestFramework.h"

class StatisticsTests : public IntegrationTestFramework
{
public:
    StatisticsTests()
        : IntegrationTestFramework()
    {}

    ~StatisticsTests() = default;
};

TEST_F(StatisticsTests, selfReplicatorWithRepetitionsInGenome)
{
    auto subGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().numRepetitions(3)).cells({CellGenomeDescription()}));
    auto mainGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().numRepetitions(2))
            .cells({
                CellGenomeDescription().cellType(ConstructorGenomeDescription().genome(subGenome)),
                CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy()),
            }));

    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).cellType(ConstructorDescription().genome(mainGenome)),
    });

    _simulationFacade->setSimulationData(data);
    auto statistics = _simulationFacade->getStatisticsRawData();

    EXPECT_EQ(1, statistics.timeline.timestep.numCells[0]);
    EXPECT_EQ(1, statistics.timeline.timestep.numSelfReplicators[0]);
    EXPECT_EQ(10, statistics.timeline.timestep.numGenomeCells[0]);
}

TEST_F(StatisticsTests, selfReplicatorWithInfiniteRepetitionsInGenome)
{
    auto subGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().infiniteRepetitions()).cells({CellGenomeDescription()}));
    auto mainGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().numRepetitions(2))
            .cells({
                CellGenomeDescription().cellType(ConstructorGenomeDescription().genome(subGenome)),
                CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy()),
            }));

    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).cellType(ConstructorDescription().genome(mainGenome)),
    });

    _simulationFacade->setSimulationData(data);
    auto statistics = _simulationFacade->getStatisticsRawData();

    EXPECT_EQ(1, statistics.timeline.timestep.numCells[0]);
    EXPECT_EQ(1, statistics.timeline.timestep.numSelfReplicators[0]);
    EXPECT_EQ(6, statistics.timeline.timestep.numGenomeCells[0]);
}

TEST_F(StatisticsTests, nonSelfReplicatorWithRepetitionsInGenome)
{
    auto subGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().numRepetitions(3)).cells({CellGenomeDescription()}));
    auto mainGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().numRepetitions(2))
            .cells({
                CellGenomeDescription().cellType(ConstructorGenomeDescription().genome(subGenome)),
            }));

    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).cellType(ConstructorDescription().genome(mainGenome)),
    });

    _simulationFacade->setSimulationData(data);
    auto statistics = _simulationFacade->getStatisticsRawData();

    EXPECT_EQ(1, statistics.timeline.timestep.numCells[0]);
    EXPECT_EQ(0, statistics.timeline.timestep.numSelfReplicators[0]);
    EXPECT_EQ(00, statistics.timeline.timestep.numGenomeCells[0]);
}
