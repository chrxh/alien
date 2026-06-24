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
        : IntegrationTestFramework({200, 200})
    {
    }

    ~BalanceTests() override = default;

    Desc createSmallCreatureSeed()
    {
        auto worldSize = toRealVector2D(_simulationFacade->getWorldSize());
        auto& numberGen = NumberGenerator::get();
        return Desc().addCreature(
            {
                ObjectDesc()
                .pos({numberGen.getRandomFloat(0.0f, worldSize.x), numberGen.getRandomFloat(0.0f, worldSize.y)})
                .type(CellDesc().headCell(true).constructor(ConstructorDesc().provideEnergy(ProvideEnergy_Free).separation(true))),
            },
            CreatureDesc().lineageId(0),
            GenomeDesc().frontAngle(225.0f).genes(
            {
                GeneDesc()

                .shape(ConstructorShape_Hexagon)
                .nodes(
                {
                    NodeDesc().cellType(SensorGenomeDesc().mode(DetectCreatureGenomeDesc().restrictToLineage(LineageRestriction_UnrelatedLineage))),
                    NodeDesc().cellType(AttackerGenomeDesc()),
                    NodeDesc()
                    .cellType(MuscleGenomeDesc().mode(DirectMovementGenomeDesc()))
                    .neuralNetwork(NeuralNetGenomeDesc().bias(0, 0.1f).connectionWeight(0, 0)),
                    NodeDesc().constructor(ConstructorGenomeDesc().separation(true)),
                    NodeDesc().cellType(DigestorGenomeDesc()),
                    NodeDesc()
                    .cellType(MuscleGenomeDesc().mode(DirectMovementGenomeDesc()))
                    .neuralNetwork(NeuralNetGenomeDesc().bias(0, 0.1f).connectionWeight(0, 0)),
                    NodeDesc().cellType(AttackerGenomeDesc()),
                }),
            }));
    }

    enum class DigestionCapability
    {
        Low,
        High
    };

    Desc createLargeCreatureSeed(const DigestionCapability& digestionCapability)
    {
        auto worldSize = toRealVector2D(_simulationFacade->getWorldSize());
        auto& numberGen = NumberGenerator::get();
        auto highDigestion = digestionCapability == DigestionCapability::High;
        auto obligatoryDigestor = DigestorGenomeDesc().rawEnergyConductivity(0.9f);
        auto optionalDigestor = highDigestion ? CellTypeGenomeDesc(DigestorGenomeDesc()) : CellTypeGenomeDesc(BaseGenomeDesc());
        return Desc().addCreature(
            {
                ObjectDesc()
                .pos({numberGen.getRandomFloat(0.0f, worldSize.x), numberGen.getRandomFloat(0.0f, worldSize.y)})
                .type(CellDesc().headCell(true).constructor(ConstructorDesc().provideEnergy(ProvideEnergy_Free).separation(true))),
            },
            CreatureDesc().lineageId(1),
            GenomeDesc().frontAngle(225.0f).genes(
            {
                GeneDesc()

                .shape(ConstructorShape_Hexagon)
                .nodes(
                {
                    NodeDesc().cellType(AttackerGenomeDesc()),
                    NodeDesc().cellType(SensorGenomeDesc().mode(DetectCreatureGenomeDesc().restrictToLineage(LineageRestriction_UnrelatedLineage))),
                    NodeDesc().cellType(obligatoryDigestor),
                    NodeDesc().cellType(optionalDigestor),
                    NodeDesc().cellType(obligatoryDigestor),
                    NodeDesc().cellType(obligatoryDigestor),
                    NodeDesc().cellType(SensorGenomeDesc().mode(DetectCreatureGenomeDesc().restrictToLineage(LineageRestriction_UnrelatedLineage))),
                    NodeDesc().cellType(AttackerGenomeDesc()),
                    NodeDesc().cellType(SensorGenomeDesc().mode(DetectCreatureGenomeDesc().restrictToLineage(LineageRestriction_UnrelatedLineage))),
                    NodeDesc().cellType(AttackerGenomeDesc()),
                    NodeDesc().cellType(obligatoryDigestor),
                    NodeDesc()
                    .cellType(MuscleGenomeDesc().mode(DirectMovementGenomeDesc()))
                    .neuralNetwork(NeuralNetGenomeDesc().bias(0, 0.1f).connectionWeight(0, 0)),
                    NodeDesc().cellType(optionalDigestor),
                    NodeDesc().cellType(optionalDigestor),
                    NodeDesc().cellType(optionalDigestor).constructor(ConstructorGenomeDesc().separation(true)),
                    NodeDesc().cellType(optionalDigestor),
                    NodeDesc().cellType(optionalDigestor),
                    NodeDesc()
                    .cellType(MuscleGenomeDesc().mode(DirectMovementGenomeDesc()))
                    .neuralNetwork(NeuralNetGenomeDesc().bias(0, 0.1f).connectionWeight(0, 0)),
                    NodeDesc().cellType(AttackerGenomeDesc()),
                }),
            }));
    }
};

// Test that small creatures dominates if large creatures have few digestion capabilities
TEST_F(BalanceTests, longRunning_smallCreatures_vs_largeCreatures_fewDigestionCapabilities)
{
    _parameters.attackerRadius.value[0] = 3.0f;
    _parameters.muscleMovementAcceleration = {ColorVector<float>::uniform(3.0f)};
    _simulationFacade->setSimulationParameters(_parameters);

    Desc data;
    for (int i = 0; i < 300; ++i) {
        data.add(createSmallCreatureSeed());
    }
    for (int i = 0; i < 15; ++i) {
        data.add(createLargeCreatureSeed(DigestionCapability::Low));
    }

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(20000);
    auto actualData = _simulationFacade->getSimulationData();

    int numSmallCreatures = 0;
    int numLargeCreatures = 0;
    for (const auto& creature : actualData._creatures) {
        auto lineageId = creature._lineageId;
        if (lineageId == 0) {
            ++numSmallCreatures;
        } else if (lineageId == 1) {
            ++numLargeCreatures;
        } else {
            CHECK(false);
        }
    }

    EXPECT_GT(14, numLargeCreatures);
    EXPECT_LT(200, numSmallCreatures);
}


// Test that large creatures dominates if large creatures have high digestion capabilities
TEST_F(BalanceTests, longRunning_smallCreatures_vs_largeCreatures_highDigestionCapabilities)
{
    _parameters.attackerRadius.value[0] = 3.0f;
    _parameters.muscleMovementAcceleration = {ColorVector<float>::uniform(3.0f)};
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

    int numSmallCreatures = 0;
    int numLargeCreatures = 0;
    for (const auto& creature : actualData._creatures) {
        auto lineageId = creature._lineageId;
        if (lineageId == 0) {
            ++numSmallCreatures;
        } else if (lineageId == 1) {
            ++numLargeCreatures;
        } else {
            CHECK(false);
        }
    }
    EXPECT_LT(20, numLargeCreatures);
}