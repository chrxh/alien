#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/StatisticsRawData.h>

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
    //auto subGenome = GenomeDescConverterService::get().convertDescriptionToBytes(
    //    GenomeDesc().header(GenomeHeaderDescription().numRepetitions(3)).objects({CellGenomeDesc()}));
    //auto mainGenome = GenomeDescConverterService::get().convertDescriptionToBytes(
    //    GenomeDesc()
    //        .header(GenomeHeaderDescription().numRepetitions(2))
    //        .objects({
    //            CellGenomeDesc().constructor(ConstructorGenomeDesc().genome(subGenome)),
    //            CellGenomeDesc().constructor(ConstructorGenomeDesc().makeSelfCopy()),
    //        }));

    //Desc data;
    //data.objects() = {
    //    ObjectDesc().id(1).type(CellDesc().constructor(ConstructorDesc()/*.genome(mainGenome)*/)),
    //});

    //_simulationFacade->setSimulationData(data);
    //auto statistics = _simulationFacade->getStatisticsRawData();

    //EXPECT_EQ(1, statistics.timeline.timestep.numObjects[0]);
    //EXPECT_EQ(1, statistics.timeline.timestep.numSelfReplicators[0]);
    //EXPECT_EQ(10, statistics.timeline.timestep.numGenomeCells[0]);
}

TEST_F(StatisticsTests, selfReplicatorWithInfiniteRepetitionsInGenome)
{
    //auto subGenome = GenomeDescConverterService::get().convertDescriptionToBytes(
    //    GenomeDesc().header(GenomeHeaderDescription().infiniteRepetitions()).objects({CellGenomeDesc()}));
    //auto mainGenome = GenomeDescConverterService::get().convertDescriptionToBytes(
    //    GenomeDesc()
    //        .header(GenomeHeaderDescription().numRepetitions(2))
    //        .objects({
    //            CellGenomeDesc().constructor(ConstructorGenomeDesc().genome(subGenome)),
    //            CellGenomeDesc().constructor(ConstructorGenomeDesc().makeSelfCopy()),
    //        }));

    //Desc data;
    //data.objects() = {
    //    ObjectDesc().id(1).type(CellDesc().constructor(ConstructorDesc()/*.genome(mainGenome)*/)),
    //});

    //_simulationFacade->setSimulationData(data);
    //auto statistics = _simulationFacade->getStatisticsRawData();

    //EXPECT_EQ(1, statistics.timeline.timestep.numObjects[0]);
    //EXPECT_EQ(1, statistics.timeline.timestep.numSelfReplicators[0]);
    //EXPECT_EQ(6, statistics.timeline.timestep.numGenomeCells[0]);
}

TEST_F(StatisticsTests, nonSelfReplicatorWithRepetitionsInGenome)
{
    //auto subGenome = GenomeDescConverterService::get().convertDescriptionToBytes(
    //    GenomeDesc().header(GenomeHeaderDescription().numRepetitions(3)).objects({CellGenomeDesc()}));
    //auto mainGenome = GenomeDescConverterService::get().convertDescriptionToBytes(
    //    GenomeDesc()
    //        .header(GenomeHeaderDescription().numRepetitions(2))
    //        .objects({
    //            CellGenomeDesc().constructor(ConstructorGenomeDesc().genome(subGenome)),
    //        }));

    //Desc data;
    //data.objects() = {
    //    ObjectDesc().id(1).type(CellDesc().constructor(ConstructorDesc()/*.genome(mainGenome)*/)),
    //});

    //_simulationFacade->setSimulationData(data);
    //auto statistics = _simulationFacade->getStatisticsRawData();

    //EXPECT_EQ(1, statistics.timeline.timestep.numObjects[0]);
    //EXPECT_EQ(0, statistics.timeline.timestep.numSelfReplicators[0]);
    //EXPECT_EQ(00, statistics.timeline.timestep.numGenomeCells[0]);
}
