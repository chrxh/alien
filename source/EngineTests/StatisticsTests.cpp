#include <algorithm>
#include <stdexcept>

#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class StatisticsTests : public MutationTestsBase
{};

TEST_F(StatisticsTests, basicCounts)
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
    EXPECT_EQ(2u, entry42->sumCreatureCells);
    EXPECT_EQ(3u, entry42->sumCreatureGenerations);
    EXPECT_GT(entry42->sumCreatureEnergy, 0.0);
    EXPECT_EQ((1u << 2) | (1u << 5), entry42->colorBitset);  // Derived from genome node colors, not cell colors

    auto const* entry43 = findEntry(43);
    ASSERT_TRUE(entry43 != nullptr);
    EXPECT_EQ(1, entry43->numCreatures);
    EXPECT_EQ(1, entry43->numGenomes);
    EXPECT_EQ(1u, entry43->sumCreatureCells);
    EXPECT_EQ(5u, entry43->sumCreatureGenerations);
    EXPECT_GT(entry43->sumCreatureEnergy, 0.0);
}

TEST_F(StatisticsTests, objectStatistics)
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

    // The total energy covers free cells, solids and energy particles, not just the cells belonging to creatures
    ASSERT_EQ(1, entries.lineageEntries.size());
    auto creatureEnergy = entries.lineageEntries.front().sumCreatureEnergy;
    EXPECT_GT(creatureEnergy, 0.0);
    EXPECT_DOUBLE_EQ(creatureEnergy + 200.0 + 400.0 + 300.0, entries.objectStatistics.totalInternalEnergy);
}

TEST_F(StatisticsTests, representativeCell)
{
    auto data = Desc()
                    .addCreature({ObjectDesc().id(1), ObjectDesc().id(2).pos({1.0f, 0.0f})}, CreatureDesc().lineageId(42).generation(3), GenomeDesc())
                    .addCreature({ObjectDesc().id(3).pos({10.0f, 10.0f})}, CreatureDesc().lineageId(42).generation(5), GenomeDesc());

    _simulationFacade->setSimulationData(data);

    auto entries = _simulationFacade->getStatisticsEntry();
    ASSERT_EQ(1, entries.lineageEntries.size());
    EXPECT_EQ(3u, entries.lineageEntries.front().representativeCellId);  // Cell of the creature with the highest generation
}

TEST_F(StatisticsTests, genomeNodesAndMutationRates)
{
    auto genome = createTestGenome();
    uint64_t numNodes = 0;
    for (auto const& gene : genome._genes) {
        numNodes += gene._nodes.size();
    }
    genome._mutationRates._voidMutation = VoidMutationDesc().nodeProbability(0.21f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc().lineageId(42), genome);
    _simulationFacade->setSimulationData(data);

    auto entries = _simulationFacade->getStatisticsEntry();
    ASSERT_EQ(1, entries.lineageEntries.size());
    auto const& entry = entries.lineageEntries.front();
    EXPECT_EQ(1u, entry.numGenomes);
    EXPECT_EQ(numNodes, entry.sumGenomeNodes);
    EXPECT_NEAR(0.01, entry.sumMutationRates, 1e-6);  // Mean over 21 probability values: 0.21 / 21
}

TEST_F(StatisticsTests, accumulatedMutations)
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

// The accumulated counters survive across tests because the simulation is shared within the test suite,
// so the following tests check the deltas caused by their own time steps
namespace
{
    LineageStatisticsEntry const& findLineageEntry(StatisticsEntry const& entries, uint32_t lineageId)
    {
        auto result = std::ranges::find_if(entries.lineageEntries, [lineageId](auto const& entry) { return entry.lineageId == lineageId; });
        if (result == entries.lineageEntries.end()) {
            throw std::runtime_error("Lineage entry not found: " + std::to_string(lineageId));
        }
        return *result;
    }
}

TEST_F(StatisticsTests, accumulatedAttackedEnergy)
{
    auto parameters = _parameters;
    parameters.attackerStrength.value = 0.1f;
    for (int i = 0; i < MAX_COLORS; ++i) {
        parameters.attackerEnergyCost.baseValue[i] = 0;
        parameters.attackerRadius.value[i] = 3.5f;
    }
    _simulationFacade->setSimulationParameters(parameters);

    NeuralNetDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    SensorLastMatchDesc lastMatch;
    lastMatch._creatureIdPart = 2 & 0xffff;
    lastMatch._pos = RealVector2D{100.0f, 103.0f};

    auto data = Desc()
                    .addCreature(
                        {ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(AttackerDesc().mode(AttackCreatureDesc())).neuralNetwork(nn)),
                         ObjectDesc()
                             .id(2)
                             .pos({101.0f, 100.0f})
                             .type(CellDesc().cellType(SensorDesc().autoTrigger(false).mode(DetectCreatureDesc()).lastMatch(lastMatch)))},
                        CreatureDesc().id(1).lineageId(42))
                    .addCreature({ObjectDesc().id(100).pos({100.0f, 103.0f}).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(2).lineageId(43));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    auto initialEntries = _simulationFacade->getStatisticsEntry();
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);
    auto entries = _simulationFacade->getStatisticsEntry();

    auto const& initial42 = findLineageEntry(initialEntries, 42);
    auto const& initial43 = findLineageEntry(initialEntries, 43);
    auto const& entry42 = findLineageEntry(entries, 42);
    auto const& entry43 = findLineageEntry(entries, 43);
    EXPECT_GT(entry42.totalAttackedEnergy, initial42.totalAttackedEnergy);
    EXPECT_DOUBLE_EQ(initial42.totalMuscleActivity, entry42.totalMuscleActivity);
    EXPECT_DOUBLE_EQ(initial43.totalAttackedEnergy, entry43.totalAttackedEnergy);
}

TEST_F(StatisticsTests, accumulatedMuscleActivity)
{
    NeuralNetDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}),
            ObjectDesc()
                .id(2)
                .pos({11.0f, 10.0f})
                .type(CellDesc().frontAngle(0.0f).headCell(true).cellType(MuscleDesc().mode(DirectMovementDesc())).neuralNetwork(nn)),
            ObjectDesc().id(3).pos({12.0f, 10.0f}),
        },
        CreatureDesc().id(1).lineageId(42));
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    auto initialEntries = _simulationFacade->getStatisticsEntry();
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);
    auto entries = _simulationFacade->getStatisticsEntry();

    auto const& initial42 = findLineageEntry(initialEntries, 42);
    auto const& entry42 = findLineageEntry(entries, 42);
    EXPECT_GT(entry42.totalMuscleActivity, initial42.totalMuscleActivity);
    EXPECT_DOUBLE_EQ(initial42.totalAttackedEnergy, entry42.totalAttackedEnergy);
}
