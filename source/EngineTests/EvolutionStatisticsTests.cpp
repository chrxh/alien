#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/StatisticsRawData.h>

#include "MutationTestsBase.h"

class EvolutionStatisticsTests : public MutationTestsBase
{};

TEST_F(EvolutionStatisticsTests, basicCounts)
{
    auto data = Desc()
                    .addCreature({ObjectDesc().id(1), ObjectDesc().id(2).pos({1.0f, 0.0f})}, CreatureDesc().lineageId(42).generation(3), GenomeDesc())
                    .addCreature({ObjectDesc().id(3).pos({10.0f, 10.0f})}, CreatureDesc().lineageId(43).generation(5), GenomeDesc());

    _simulationFacade->setSimulationData(data);

    auto statistics = _simulationFacade->getStatisticsRawData();
    auto const& evolution = statistics.timeline.timestep.evolution;
    EXPECT_EQ(2, evolution.numCreatures);
    EXPECT_EQ(2, evolution.numGenomes);
    EXPECT_EQ(2, evolution.numActiveLineages);
    EXPECT_EQ(3, evolution.numCellObjects);
    EXPECT_EQ(0, evolution.numSolidObjects);
    EXPECT_EQ(0, evolution.numFluidObjects);
    EXPECT_EQ(0, evolution.lineageMapOverflow);
    EXPECT_FLOAT_EQ(3.0f, evolution.sumCreatureCells);
    EXPECT_FLOAT_EQ(8.0f, evolution.sumCreatureGenerations);
    EXPECT_GT(evolution.sumCreatureEnergy, 0.0f);

    auto lineageStatistics = _simulationFacade->getRawLineageStatistics();
    ASSERT_EQ(2, lineageStatistics.entries.size());
    auto findEntry = [&](uint32_t lineageId) -> LineageStatisticsEntry const* {
        for (auto const& entry : lineageStatistics.entries) {
            if (entry.lineageId == lineageId) {
                return &entry;
            }
        }
        return nullptr;
    };
    auto const* entry42 = findEntry(42);
    ASSERT_TRUE(entry42 != nullptr);
    EXPECT_EQ(1, entry42->numCreatures);
    EXPECT_EQ(1, entry42->numGenomes);
    EXPECT_FLOAT_EQ(2.0f, entry42->sumCreatureCells);
    EXPECT_FLOAT_EQ(3.0f, entry42->sumCreatureGenerations);
    EXPECT_GT(entry42->sumCreatureEnergy, 0.0f);
    EXPECT_NE(0, entry42->colorBitset);

    auto const* entry43 = findEntry(43);
    ASSERT_TRUE(entry43 != nullptr);
    EXPECT_EQ(1, entry43->numCreatures);
    EXPECT_FLOAT_EQ(1.0f, entry43->sumCreatureCells);
    EXPECT_FLOAT_EQ(5.0f, entry43->sumCreatureGenerations);
}

TEST_F(EvolutionStatisticsTests, genomeNodesAndMutationRates)
{
    auto genome = createTestGenome();
    auto numNodes = 0;
    for (auto const& gene : genome._genes) {
        numNodes += toInt(gene._nodes.size());
    }
    genome._mutationRates._voidMutation = VoidMutationDesc().nodeProbability(0.21f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc().lineageId(42), genome);
    _simulationFacade->setSimulationData(data);

    auto statistics = _simulationFacade->getStatisticsRawData();
    auto const& evolution = statistics.timeline.timestep.evolution;
    EXPECT_EQ(1, evolution.numGenomes);
    EXPECT_FLOAT_EQ(toFloat(numNodes), evolution.sumGenomeNodes);
    EXPECT_FLOAT_EQ(0.01f, evolution.sumMutationRates);  //mean over 21 probability values: 0.21 / 21
}

TEST_F(EvolutionStatisticsTests, accumulatedMutations)
{
    auto genome = createTestGenome();
    genome._mutationRates._neuronMutations[0] =
        NeuronMutationDesc().nodeProbability(1.0f).weightChangeSigma(1.0f).biasChangeSigma(1.0f).actfnChangeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc().lineageId(42), genome);

    auto parameters = _parameters;
    parameters.newLineageThreshold.value = 1.0e30f;
    _simulationFacade->setSimulationParameters(parameters);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }
    _simulationFacade->calcTimesteps(1);

    auto statistics = _simulationFacade->getStatisticsRawData();
    EXPECT_GT(statistics.timeline.timestep.evolution.sumAccumulatedMutations, 0.0f);
    EXPECT_GT(statistics.timeline.accumulated.totalMutations, 0.0);

    auto lineageStatistics = _simulationFacade->getRawLineageStatistics();
    ASSERT_EQ(1, lineageStatistics.entries.size());
    EXPECT_EQ(42, lineageStatistics.entries.front().lineageId);
    EXPECT_GT(lineageStatistics.entries.front().totalMutations, 0.0f);
}
