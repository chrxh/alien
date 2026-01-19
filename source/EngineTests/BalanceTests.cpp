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

    Desc createSmallCreatureSeed()
    {
        auto worldSize = toRealVector2D(_simulationFacade->getWorldSize());
        auto& numberGen = NumberGenerator::get();
        return Desc().addCreature(
            {
                ObjectDesc().pos({numberGen.getRandomFloat(0.0f, worldSize.x), numberGen.getRandomFloat(0.0f, worldSize.y)}).type(CellDesc().cellType(ConstructorDesc().provideEnergy(ProvideEnergy_FreeGeneration))),
            },
            CreatureDesc().lineageId(0),
            GenomeDesc().genes({
                GeneDesc()
                    .separation(true)
                    .shape(ConstructorShape_Hexagon)
                    .nodes({
                        NodeDesc().cellType(DigestorGenomeDesc()),
                        NodeDesc().cellType(AttackerGenomeDesc()),
                        NodeDesc().cellType(MuscleGenomeDesc().mode(DirectMovementGenomeDesc())),
                        NodeDesc().cellType(ConstructorGenomeDesc()),
                        NodeDesc().cellType(
                            SensorGenomeDesc().mode(DetectCreatureGenomeDesc().restrictToLineage(LineageRestriction_OtherLineage))),
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
        auto rawEnergyConductivity = digestionCapability == DigestionCapability::Low ? 1.0f : 0.0f;
        return Desc().addCreature(
            {
                ObjectDesc().pos({numberGen.getRandomFloat(0.0f, worldSize.x), numberGen.getRandomFloat(0.0f, worldSize.y)}).type(CellDesc().cellType(ConstructorDesc().provideEnergy(ProvideEnergy_FreeGeneration))),
            },
            CreatureDesc().lineageId(1),
            GenomeDesc().genes({
                GeneDesc()
                    .separation(true)
                    .shape(ConstructorShape_Hexagon)
                    .nodes({
                        NodeDesc().cellType(GeneratorGenomeDesc().autoTriggerInterval(15)),
                        NodeDesc().cellType(DigestorGenomeDesc()),
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDesc().cellType(AttackerGenomeDesc()),
                        NodeDesc().cellType(AttackerGenomeDesc()),
                        NodeDesc().cellType(AttackerGenomeDesc()),
                        NodeDesc().cellType(AttackerGenomeDesc()),
                        NodeDesc().cellType(MuscleGenomeDesc().mode(DirectMovementGenomeDesc())),
                        NodeDesc().cellType(MuscleGenomeDesc().mode(DirectMovementGenomeDesc())),
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDesc().cellType(ConstructorGenomeDesc()),
                        NodeDesc().cellType(DigestorGenomeDesc().rawEnergyConductivity(rawEnergyConductivity)),
                        NodeDesc().cellType(MuscleGenomeDesc().mode(DirectMovementGenomeDesc())),
                        NodeDesc().cellType(MuscleGenomeDesc().mode(DirectMovementGenomeDesc())),
                        NodeDesc().cellType(AttackerGenomeDesc()),
                    }),
            }));
    }
};

// Test that small creatures dominates if large creatures have few digestion capabilities
TEST_F(BalanceTests, longRunning_smallCreatures_vs_largeCreatures_fewDigestionCapabilities)
{
    _parameters.attackerRadius.value[0] = 4.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    Desc data;
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

    Desc data;
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