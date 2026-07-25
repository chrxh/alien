#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class EvolutionStatisticsTests : public MutationTestsBase
{};

TEST_F(EvolutionStatisticsTests, basicCounts)
{
    auto genome42 = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().color(2), NodeDesc().color(5)})});
    auto data = Desc()
                    .addCreature({ObjectDesc().id(1), ObjectDesc().id(2).pos({1.0f, 0.0f})}, CreatureDesc().lineageId(42).generation(3), genome42)
                    .addCreature({ObjectDesc().id(3).pos({10.0f, 10.0f})}, CreatureDesc().lineageId(43).generation(5), GenomeDesc());

    _simulationFacade->setSimulationData(data);

    auto entries = _simulationFacade->getStatisticsEntry();
    EXPECT_EQ(3u, entries.objectStatistics.numCellObjects);
    EXPECT_EQ(0u, entries.objectStatistics.numSolidObjects);
    EXPECT_EQ(0u, entries.objectStatistics.numFluidObjects);
    EXPECT_EQ(0u, entries.objectStatistics.numFreeCellObjects);

    ASSERT_EQ(2, entries.lineageEntries.size());
    auto findEntry = [&](uint32_t lineageId) -> LineageStatisticsEntry const* {
        for (auto const& entry : entries.lineageEntries) {
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
    EXPECT_EQ((1u << 2) | (1u << 5), entry42->colorBitset);  //derived from genome node colors, not cell colors

    auto const* entry43 = findEntry(43);
    ASSERT_TRUE(entry43 != nullptr);
    EXPECT_EQ(1, entry43->numCreatures);
    EXPECT_EQ(1, entry43->numGenomes);
    EXPECT_FLOAT_EQ(1.0f, entry43->sumCreatureCells);
    EXPECT_FLOAT_EQ(5.0f, entry43->sumCreatureGenerations);
    EXPECT_GT(entry43->sumCreatureEnergy, 0.0f);
}

TEST_F(EvolutionStatisticsTests, objectStatistics)
{
    auto data = Desc()
                    .addCreature({ObjectDesc().id(1)}, CreatureDesc().lineageId(42), GenomeDesc())
                    .addObjects(
                        {ObjectDesc().id(2).pos({10.0f, 0.0f}).type(FreeCellDesc().energy(200.0f)),
                         ObjectDesc().id(3).pos({20.0f, 0.0f}).type(SolidDesc().energy(400.0f))})
                    .energies({EnergyDesc().id(4).pos({30.0f, 0.0f}).energy(300.0f)});

    _simulationFacade->setSimulationData(data);

    auto entries = _simulationFacade->getStatisticsEntry();
    EXPECT_EQ(1u, entries.objectStatistics.numCellObjects);
    EXPECT_EQ(1u, entries.objectStatistics.numFreeCellObjects);
    EXPECT_EQ(1u, entries.objectStatistics.numSolidObjects);
    EXPECT_EQ(1u, entries.objectStatistics.numEnergyParticles);

    //the total energy covers free cells, solids and energy particles, not just the cells belonging to creatures
    ASSERT_EQ(1, entries.lineageEntries.size());
    auto creatureEnergy = toDouble(entries.lineageEntries.front().sumCreatureEnergy);
    EXPECT_GT(creatureEnergy, 0.0);
    EXPECT_DOUBLE_EQ(creatureEnergy + 200.0 + 400.0 + 300.0, entries.objectStatistics.totalInternalEnergy);
}

TEST_F(EvolutionStatisticsTests, representativeCell)
{
    auto data = Desc()
                    .addCreature({ObjectDesc().id(1), ObjectDesc().id(2).pos({1.0f, 0.0f})}, CreatureDesc().lineageId(42).generation(3), GenomeDesc())
                    .addCreature({ObjectDesc().id(3).pos({10.0f, 10.0f})}, CreatureDesc().lineageId(42).generation(5), GenomeDesc());

    _simulationFacade->setSimulationData(data);

    auto entries = _simulationFacade->getStatisticsEntry();
    ASSERT_EQ(1, entries.lineageEntries.size());
    EXPECT_EQ(3u, entries.lineageEntries.front().representativeCellId);  //cell of the creature with the highest generation
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

    auto entries = _simulationFacade->getStatisticsEntry();
    ASSERT_EQ(1, entries.lineageEntries.size());
    auto const& entry = entries.lineageEntries.front();
    EXPECT_EQ(1u, entry.numGenomes);
    EXPECT_FLOAT_EQ(toFloat(numNodes), entry.sumGenomeNodes);
    EXPECT_FLOAT_EQ(0.01f, entry.sumMutationRates);  //mean over 21 probability values: 0.21 / 21
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

    auto entries = _simulationFacade->getStatisticsEntry();
    ASSERT_EQ(1, entries.lineageEntries.size());
    EXPECT_EQ(42, entries.lineageEntries.front().lineageId);
    EXPECT_GT(entries.lineageEntries.front().totalMutations, 0.0f);
}
