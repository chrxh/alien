#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class BalanceTests : public IntegrationTestFramework
{
public:
    BalanceTests()
        : IntegrationTestFramework({100, 100})
    {}

    ~BalanceTests() = default;

    Description createSmallCreatureSeed()
    {
        auto worldSize = toRealVector2D(_simulationFacade->getWorldSize());
        auto& numberGen = NumberGenerator::get();
        return Description().addCreature(
            {
                ObjectDescription()
                    .pos({numberGen.getRandomFloat(0.0f, worldSize.x), numberGen.getRandomFloat(0.0f, worldSize.y)})
                    .cellType(ConstructorDescription().provideEnergy(ProvideEnergy_FreeGeneration)),
            },
            CreatureDescription().lineageId(0),
            GenomeDescription().genes({
                GeneDescription()
                    .separation(true)
                    .shape(ConstructorShape_Hexagon)
                    .nodes({
                        NodeDescription().cellType(DigestorGenomeDescription()),
                        NodeDescription().cellType(AttackerGenomeDescription()),
                        NodeDescription().cellType(MuscleGenomeDescription().mode(DirectMovementGenomeDescription())),
                        NodeDescription().cellType(ConstructorGenomeDescription()),
                        NodeDescription().cellType(
                            SensorGenomeDescription().mode(DetectCreatureGenomeDescription().restrictToLineage(LineageRestriction_OtherLineage))),
                        NodeDescription().cellType(MuscleGenomeDescription().mode(DirectMovementGenomeDescription())),
                        NodeDescription().cellType(AttackerGenomeDescription()),
                    }),
            }));
    }

    enum class DigestionCapability
    {
        Low,
        High
    };
    Description createLargeCreatureSeed(DigestionCapability const& digestionCapability)
    {
        auto worldSize = toRealVector2D(_simulationFacade->getWorldSize());
        auto& numberGen = NumberGenerator::get();
        auto rawEnergyConductivity = digestionCapability == DigestionCapability::Low ? 1.0f : 0.0f;
        return Description().addCreature(
            {
                ObjectDescription()
                    .pos({numberGen.getRandomFloat(0.0f, worldSize.x), numberGen.getRandomFloat(0.0f, worldSize.y)})
                    .cellType(ConstructorDescription().provideEnergy(ProvideEnergy_FreeGeneration)),
            },
            CreatureDescription().lineageId(1),
            GenomeDescription().genes({
                GeneDescription()
                    .separation(true)
                    .shape(ConstructorShape_Hexagon)
                    .nodes({
                        NodeDescription().cellType(GeneratorGenomeDescription().autoTriggerInterval(15)),
                        NodeDescription().cellType(DigestorGenomeDescription()),
                        NodeDescription().cellType(DigestorGenomeDescription().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDescription().cellType(DigestorGenomeDescription().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDescription().cellType(DigestorGenomeDescription().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDescription().cellType(DigestorGenomeDescription().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDescription().cellType(DigestorGenomeDescription().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDescription().cellType(AttackerGenomeDescription()),
                        NodeDescription().cellType(AttackerGenomeDescription()),
                        NodeDescription().cellType(AttackerGenomeDescription()),
                        NodeDescription().cellType(AttackerGenomeDescription()),
                        NodeDescription().cellType(MuscleGenomeDescription().mode(DirectMovementGenomeDescription())),
                        NodeDescription().cellType(MuscleGenomeDescription().mode(DirectMovementGenomeDescription())),
                        NodeDescription().cellType(DigestorGenomeDescription().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDescription().cellType(ConstructorGenomeDescription()),
                        NodeDescription().cellType(DigestorGenomeDescription().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDescription().cellType(MuscleGenomeDescription().mode(DirectMovementGenomeDescription())),
                        NodeDescription().cellType(MuscleGenomeDescription().mode(DirectMovementGenomeDescription())),
                        NodeDescription().cellType(AttackerGenomeDescription()),
                    }),
            }));
    }
};

// Test that small creatures dominates if large creatures have few digestion capabilities
TEST_F(BalanceTests, longRunning_smallCreatures_vs_largeCreatures_fewDigestionCapabilities)
{
    _parameters.attackerRadius.value[0] = 4.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    Description data;
    for (int i = 0; i < 300; ++i) {
        data.add(createSmallCreatureSeed());
    }
    for (int i = 0; i < 8; ++i) {
        data.add(createLargeCreatureSeed(DigestionCapability::Low));
    }

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(100000);
    auto actualData = _simulationFacade->getSimulationData();

    int numSmallCreatures = 0;
    int numLargeCreatures = 0;
    for (auto const& creature : actualData._creatures) {
        if (creature._lineageId == 0) {
            ++numSmallCreatures;
        } else if (creature._lineageId == 1) {
            ++numLargeCreatures;
        } else {
            CHECK(false);
        }
    }
    EXPECT_GT(3, numLargeCreatures);
    EXPECT_LT(200, numSmallCreatures);
}


// Test that large creatures dominates if large creatures have high digestion capabilities
TEST_F(BalanceTests, longRunning_smallCreatures_vs_largeCreatures_highDigestionCapabilities)
{
    _parameters.attackerRadius.value[0] = 4.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    Description data;
    for (int i = 0; i < 300; ++i) {
        data.add(createSmallCreatureSeed());
    }
    for (int i = 0; i < 8; ++i) {
        data.add(createLargeCreatureSeed(DigestionCapability::High));
    }

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(100000);
    auto actualData = _simulationFacade->getSimulationData();

    int numSmallCreatures = 0;
    int numLargeCreatures = 0;
    for (auto const& creature : actualData._creatures) {
        if (creature._lineageId == 0) {
            ++numSmallCreatures;
        } else if (creature._lineageId == 1) {
            ++numLargeCreatures;
        } else {
            CHECK(false);
        }
    }
    EXPECT_LT(15, numLargeCreatures);
}