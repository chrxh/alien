#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Descs.h>
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

    ContentDesc createSmallCreatureSeed()
    {
        auto worldSize = toRealVector2D(_simulationFacade->getWorldSize());
        auto& numberGen = NumberGenerator::get();
        return ContentDesc().addCreature(
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

    ContentDesc createLargeCreatureSeed(const DigestionCapability& digestionCapability)
    {
        auto worldSize = toRealVector2D(_simulationFacade->getWorldSize());
        auto& numberGen = NumberGenerator::get();
        auto highDigestion = digestionCapability == DigestionCapability::High;
        auto obligatoryDigestor = DigestorGenomeDesc().rawEnergyConductivity(0.9f);
        auto optionalDigestor = highDigestion ? CellTypeGenomeDesc(DigestorGenomeDesc()) : CellTypeGenomeDesc(BaseGenomeDesc());
        return ContentDesc().addCreature(
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

    struct LineageStats
    {
        int numSmallCreatures = 0;
        int numSmallCells = 0;
        int numLargeCreatures = 0;
        int numLargeCells = 0;
    };

    LineageStats calcLineageStats()
    {
        auto data = _simulationFacade->getSimulationData();

        LineageStats result;
        for (const auto& creature : data._creatures) {
            if (creature._lineageId == 0) {
                ++result.numSmallCreatures;
                result.numSmallCells += creature._numCells;
            } else if (creature._lineageId == 1) {
                ++result.numLargeCreatures;
                result.numLargeCells += creature._numCells;
            } else {
                CHECK(false);
            }
        }
        return result;
    }
};

// Test that the large creatures die out if they have few digestion capabilities
TEST_F(BalanceTests, longRunning_smallCreatures_vs_largeCreatures_fewDigestionCapabilities)
{
    _parameters.attackerRadius.value[0] = 3.0f;
    _parameters.muscleMovementAcceleration = {ColorVector<float>::uniform(3.0f)};
    _simulationFacade->setSimulationParameters(_parameters);

    ContentDesc data;
    for (int i = 0; i < 300; ++i) {
        data.add(createSmallCreatureSeed());
    }
    for (int i = 0; i < 15; ++i) {
        data.add(createLargeCreatureSeed(DigestionCapability::Low));
    }

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(25000);


    auto stats = calcLineageStats();
    EXPECT_GT(90, stats.numLargeCells);
    EXPECT_LT(2000, stats.numSmallCells);
}


// Test that the large creatures expand if they have high digestion capabilities
TEST_F(BalanceTests, longRunning_smallCreatures_vs_largeCreatures_highDigestionCapabilities)
{
    _parameters.attackerRadius.value[0] = 3.0f;
    _parameters.muscleMovementAcceleration = {ColorVector<float>::uniform(3.0f)};
    _simulationFacade->setSimulationParameters(_parameters);

    ContentDesc data;
    for (int i = 0; i < 300; ++i) {
        data.add(createSmallCreatureSeed());
    }
    for (int i = 0; i < 16; ++i) {
        data.add(createLargeCreatureSeed(DigestionCapability::High));
    }

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(25000);

    auto stats = calcLineageStats();
    EXPECT_LT(90, stats.numLargeCells);
}