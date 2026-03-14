#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/GenomeDesc.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SimulationFacade.h>
#include <PersisterInterface/SerializerService.h>

#include "IntegrationTestFramework.h"

class BalanceTests : public IntegrationTestFramework
{
public:
    BalanceTests()
        : IntegrationTestFramework({100, 100})
    {}

    ~BalanceTests() = default;

    Desc createSmallCreatureSeed()
    {
        auto worldSize = toRealVector2D(_simulationFacade->getWorldSize());
        auto& numberGen = NumberGenerator::get();
        return Desc().addCreature(
            {
                ObjectDesc()
                    .pos({numberGen.getRandomFloat(0.0f, worldSize.x), numberGen.getRandomFloat(0.0f, worldSize.y)})
                    .type(CellDesc().constructor(ConstructorDesc().provideEnergy(ProvideEnergy_FreeGeneration))),
            },
            CreatureDesc(),
            GenomeDesc().lineageId(0).prevLineageId(0).genes({
                GeneDesc()
                    .separation(true)
                    .shape(ConstructorShape_Hexagon)
                    .nodes({
                        NodeDesc().cellType(DigestorGenomeDesc()),
                        NodeDesc().cellType(AttackerGenomeDesc()),
                        NodeDesc().cellType(MuscleGenomeDesc().mode(DirectMovementGenomeDesc())),
                        NodeDesc().constructor(ConstructorGenomeDesc()),
                        NodeDesc().cellType(
                            SensorGenomeDesc().mode(DetectCreatureGenomeDesc().restrictToLineage(LineageRestriction_UnrelatedLineage))),
                        NodeDesc().cellType(MuscleGenomeDesc().mode(DirectMovementGenomeDesc())),
                        NodeDesc().cellType(AttackerGenomeDesc()),
                    }),
            }));
    }

    enum class DigestionCapability
    {
        Low,
        High
    };
    Desc createLargeCreatureSeed(DigestionCapability const& digestionCapability)
    {
        auto worldSize = toRealVector2D(_simulationFacade->getWorldSize());
        auto& numberGen = NumberGenerator::get();
        auto rawEnergyConductivity = digestionCapability == DigestionCapability::Low ? 0.8f : 0.2f;
        return Desc().addCreature(
            {
                ObjectDesc().pos({numberGen.getRandomFloat(0.0f, worldSize.x), numberGen.getRandomFloat(0.0f, worldSize.y)}).type(CellDesc().constructor(ConstructorDesc().provideEnergy(ProvideEnergy_FreeGeneration))),
            },
            CreatureDesc(),
            GenomeDesc().lineageId(1).prevLineageId(1).genes({
                GeneDesc()
                    .separation(true)
                    .shape(ConstructorShape_Hexagon)
                    .nodes({
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDesc().cellType(DigestorGenomeDesc()),
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDesc().cellType(AttackerGenomeDesc()),
                        NodeDesc().cellType(SensorGenomeDesc().mode(DetectCreatureGenomeDesc().restrictToLineage(LineageRestriction_UnrelatedLineage))),
                        NodeDesc().cellType(AttackerGenomeDesc()),
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDesc().cellType(AttackerGenomeDesc()),
                        NodeDesc().cellType(SensorGenomeDesc().mode(DetectCreatureGenomeDesc().restrictToLineage(LineageRestriction_UnrelatedLineage))),
                        NodeDesc().cellType(AttackerGenomeDesc()),
                        NodeDesc().constructor(ConstructorGenomeDesc()),
                        NodeDesc().cellType(AttackerGenomeDesc()),
                        NodeDesc().cellType(SensorGenomeDesc().mode(DetectCreatureGenomeDesc().restrictToLineage(LineageRestriction_UnrelatedLineage))),
                        NodeDesc().cellType(AttackerGenomeDesc()),
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                    }),
            }));
    }
};

// Test that small creatures dominates if large creatures have few digestion capabilities
TEST_F(BalanceTests, longRunning_smallCreatures_vs_largeCreatures_fewDigestionCapabilities)
{
    _parameters.attackerRadius.value[0] = 3.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    Desc data;
    for (int i = 0; i < 300; ++i) {
        data.add(createSmallCreatureSeed());
    }
    for (int i = 0; i < 15; ++i) {
        data.add(createLargeCreatureSeed(DigestionCapability::Low));
    }

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(12000);
    auto actualData = _simulationFacade->getSimulationData();

    // Create a map of genomeId to lineageId
    std::unordered_map<uint64_t, int> genomeIdToLineageId;
    for (auto const& genome : actualData._genomes) {
        genomeIdToLineageId[genome._id] = genome._lineageId;
    }

    int numSmallCreatures = 0;
    int numLargeCreatures = 0;
    for (auto const& creature : actualData._creatures) {
        auto lineageId = genomeIdToLineageId.at(creature._genomeId);
        if (lineageId == 0) {
            ++numSmallCreatures;
        } else if (lineageId == 1) {
            ++numLargeCreatures;
        } else {
            CHECK(false);
        }
    }

    EXPECT_GT(10, numLargeCreatures);
    EXPECT_LT(200, numSmallCreatures);
}


// Test that large creatures dominates if large creatures have high digestion capabilities
TEST_F(BalanceTests, longRunning_smallCreatures_vs_largeCreatures_highDigestionCapabilities)
{
    _parameters.attackerRadius.value[0] = 3.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    Desc data;
    for (int i = 0; i < 300; ++i) {
        data.add(createSmallCreatureSeed());
    }
    for (int i = 0; i < 16; ++i) {
        data.add(createLargeCreatureSeed(DigestionCapability::High));
    }

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(20000);
    auto actualData = _simulationFacade->getSimulationData();

    // Create a map of genomeId to lineageId
    std::unordered_map<uint64_t, int> genomeIdToLineageId;
    for (auto const& genome : actualData._genomes) {
        genomeIdToLineageId[genome._id] = genome._lineageId;
    }

    int numSmallCreatures = 0;
    int numLargeCreatures = 0;
    for (auto const& creature : actualData._creatures) {
        auto lineageId = genomeIdToLineageId.at(creature._genomeId);
        if (lineageId == 0) {
            ++numSmallCreatures;
        } else if (lineageId == 1) {
            ++numLargeCreatures;
        } else {
            CHECK(false);
        }
    }
    EXPECT_LT(25, numLargeCreatures);
}