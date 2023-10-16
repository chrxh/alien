#include <gtest/gtest.h>

#include "Base/NumberGenerator.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/GenomeDescriptionConverter.h"
#include "EngineInterface/StatisticsData.h"

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
    auto subGenome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setNumRepetitions(3)).setCells({CellGenomeDescription()}));
    auto mainGenome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setNumRepetitions(2))
            .setCells({
                CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setGenome(subGenome)),
                CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeGenomeCopy()),
            }));

    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(mainGenome)),
    });

    _simController->setSimulationData(data);
    auto statistics = _simController->getStatistics();

    EXPECT_EQ(1, statistics.timeline.timestep.numCells[0]);
    EXPECT_EQ(1, statistics.timeline.timestep.numSelfReplicators[0]);
    EXPECT_EQ(10, statistics.timeline.timestep.numGenomeCells[0]);
}

TEST_F(StatisticsTests, selfReplicatorWithInfiniteRepetitionsInGenome)
{
    auto subGenome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setInfiniteRepetitions()).setCells({CellGenomeDescription()}));
    auto mainGenome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setNumRepetitions(2))
            .setCells({
                CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setGenome(subGenome)),
                CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeGenomeCopy()),
            }));

    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(mainGenome)),
    });

    _simController->setSimulationData(data);
    auto statistics = _simController->getStatistics();

    EXPECT_EQ(1, statistics.timeline.timestep.numCells[0]);
    EXPECT_EQ(1, statistics.timeline.timestep.numSelfReplicators[0]);
    EXPECT_EQ(6, statistics.timeline.timestep.numGenomeCells[0]);
}

TEST_F(StatisticsTests, nonSelfReplicatorWithRepetitionsInGenome)
{
    auto subGenome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setNumRepetitions(3)).setCells({CellGenomeDescription()}));
    auto mainGenome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setNumRepetitions(2))
            .setCells({
                CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setGenome(subGenome)),
            }));

    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(mainGenome)),
    });

    _simController->setSimulationData(data);
    auto statistics = _simController->getStatistics();

    EXPECT_EQ(1, statistics.timeline.timestep.numCells[0]);
    EXPECT_EQ(0, statistics.timeline.timestep.numSelfReplicators[0]);
    EXPECT_EQ(00, statistics.timeline.timestep.numGenomeCells[0]);
}
