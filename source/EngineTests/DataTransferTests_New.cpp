#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"


class DataTransferTests_New : public IntegrationTestFramework
{
public:
    DataTransferTests_New()
        : IntegrationTestFramework()
    {}

};

struct CellParameter
{
    CellType cellType;
    MuscleMode muscleMode;
};

class DataTransferTests_AllCellType_New
    : public DataTransferTests_New
    , public testing::WithParamInterface<CellParameter>
{
protected:
    CellTypeDescription createSomeCellTypeDescription(CellParameter cellParameter)
    {
        auto const& type = cellParameter.cellType;
        auto const& muscleMode = cellParameter.muscleMode;
        switch (type) {
        case CellType_Structure:
            return StructureCellDescription();
        case CellType_Free:
            return FreeCellDescription();
        case CellType_Base:
            return BaseDescription();
        case CellType_Depot:
            return DepotDescription();
        case CellType_Constructor:
            return ConstructorDescription()
                .autoTriggerInterval(7)
                .geneIndex(3)
                .constructionActivationTime(4)
                .constructionAngle(34.4f)
                .lastConstructedCellId(45ull);
        case CellType_Sensor:
            return SensorDescription().autoTriggerInterval(3).restrictToColor(5).minRange(34).maxRange(67).minDensity(0.25f).restrictToMutants(
                SensorRestrictToMutants_RestrictToLessComplexMutants);
        case CellType_Oscillator:
            return OscillatorDescription().autoTriggerInterval(27).alternationInterval(45).numPulses(23);
        case CellType_Attacker:
            return AttackerDescription();
        case CellType_Injector:
            return InjectorDescription().counter(23);
        case CellType_Muscle: {
            MuscleModeDescription muscleModeDesc;
            switch (muscleMode) {
            case MuscleMode_AutoBending:
                muscleModeDesc = AutoBendingDescription()
                                     .maxAngleDeviation(0.45f)
                                     .frontBackVelRatio(0.3f)
                                     .initialAngle(23.0f)
                                     .lastActualAngle(45.0f)
                                     .forward(false)
                                     .activation(0.3f)
                                     .activationCountdown(13)
                                     .impulseAlreadyApplied(true);
                break;
            case MuscleMode_ManualBending:
                muscleModeDesc = ManualBendingDescription()
                                     .maxAngleDeviation(0.45f)
                                     .frontBackVelRatio(0.3f)
                                     .initialAngle(23.0f)
                                     .lastActualAngle(45.0f)
                                     .lastAngleDelta(2.0f)
                                     .impulseAlreadyApplied(true);
                break;
            case MuscleMode_AngleBending:
                muscleModeDesc = AngleBendingDescription().maxAngleDeviation(0.45f).frontBackVelRatio(0.3f).initialAngle(23.0f);
                break;
            case MuscleMode_AutoCrawling:
                muscleModeDesc = AutoCrawlingDescription()
                                     .maxDistanceDeviation(0.45f)
                                     .frontBackVelRatio(0.3f)
                                     .initialDistance(0.6f)
                                     .lastActualDistance(0.9f)
                                     .forward(false)
                                     .activation(0.3f)
                                     .activationCountdown(13)
                                     .impulseAlreadyApplied(true);
                break;
            case MuscleMode_ManualCrawling:
                muscleModeDesc = ManualCrawlingDescription()
                                     .maxDistanceDeviation(0.45f)
                                     .frontBackVelRatio(0.3f)
                                     .initialDistance(0.6f)
                                     .lastActualDistance(0.9f)
                                     .lastDistanceDelta(0.4f)
                                     .impulseAlreadyApplied(true);
                break;
            case MuscleMode_DirectMovement:
                muscleModeDesc = DirectMovementDescription();
                break;
            default:
                muscleModeDesc = MuscleModeDescription();
            }
            return MuscleDescription().mode(muscleModeDesc);
        }
        case CellType_Defender:
            return DefenderDescription().mode(DefenderMode_DefendAgainstInjector);
        case CellType_Reconnector:
            return ReconnectorDescription().restrictToColor(4).restrictToMutants(ReconnectorRestrictToMutants_RestrictToMoreComplexMutants);
        case CellType_Detonator:
            return DetonatorDescription().countdown(23);
        default:
            return CellTypeDescription();
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    DataTransferTests_AllCellType_New,
    DataTransferTests_AllCellType_New,
    ::testing::Values(
        CellParameter{CellType_Structure},
        CellParameter{CellType_Free},
        CellParameter{CellType_Base},
        CellParameter{CellType_Depot},
        CellParameter{CellType_Constructor},
        CellParameter{CellType_Sensor},
        CellParameter{CellType_Oscillator},
        CellParameter{CellType_Attacker},
        CellParameter{CellType_Injector},
        CellParameter{CellType_Muscle, MuscleMode_AutoBending},
        CellParameter{CellType_Muscle, MuscleMode_ManualBending},
        CellParameter{CellType_Muscle, MuscleMode_AngleBending},
        CellParameter{CellType_Muscle, MuscleMode_AutoCrawling},
        CellParameter{CellType_Muscle, MuscleMode_ManualCrawling},
        CellParameter{CellType_Muscle, MuscleMode_DirectMovement},
        CellParameter{CellType_Defender},
        CellParameter{CellType_Reconnector},
        CellParameter{CellType_Detonator}));

TEST_P(DataTransferTests_AllCellType_New, singleCell_noGenome)
{
    auto cellParameter = GetParam();
    auto cellTypeDesc = createSomeCellTypeDescription(cellParameter);

    CollectionDescription data;
    auto cell = CellDescription()
                    .id(1)
                    .pos({2.0f, 4.0f})
                    .vel({0.5f, 1.0f})
                    .energy(100.0f)
                    .age(1)
                    .color(2)
                    .barrier(true)
                    .cellState(false)
                    .signalAndRelaxTime({1, 0, -1, 0, 0, 0, 0, 0})
                    .signalRoutingRestriction(SignalRoutingRestrictionDescription().active(true).baseAngle(23.0f).openingAngle(42.0f))
                    .cellTypeData(cellTypeDesc)
                    .metadata(CellMetadataDescription().name("Test1").description("Test2"));
    if (cellParameter.cellType != CellType_Structure && cellParameter.cellType != CellType_Free) {
        NeuralNetworkDescription nn;
        nn.weight(2, 1, 1.0f);
        cell._neuralNetwork = nn;
    }
    data.addCell(cell);

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

struct NodeParameter
{
    CellTypeGenome cellTypeGenome;
    MuscleMode muscleMode;
};

class DataTransferTests_AllCellTypeGenome_New
    : public DataTransferTests_New
    , public testing::WithParamInterface<NodeParameter>
{
protected:
    CellTypeGenomeDescription_New createSomeCellTypeGenomeDescription(NodeParameter cellParameter)
    {
        auto const& type = cellParameter.cellTypeGenome;
        auto const& muscleMode = cellParameter.muscleMode;
        switch (type) {
        case CellTypeGenome_Base:
            return BaseGenomeDescription();
        case CellTypeGenome_Depot:
            return DepotGenomeDescription();
        case CellTypeGenome_Constructor:
            return ConstructorGenomeDescription_New().autoTriggerInterval(7).constructionActivationTime(4).constructionAngle(34.4f);
        case CellTypeGenome_Sensor:
            return SensorGenomeDescription_New().autoTriggerInterval(3).restrictToColor(5).minRange(34).maxRange(67).minDensity(0.25f).restrictToMutants(
                SensorRestrictToMutants_RestrictToLessComplexMutants);
        case CellTypeGenome_Oscillator:
            return OscillatorGenomeDescription().autoTriggerInterval(27).pulseType(OscillatorPulseType_Alternation).alternationInterval(45);
        case CellTypeGenome_Attacker:
            return AttackerGenomeDescription();
        case CellTypeGenome_Injector:
            return InjectorGenomeDescription_New();
        case CellTypeGenome_Muscle: {
            MuscleModeGenomeDescription muscleModeDesc;
            switch (muscleMode) {
            case MuscleMode_AutoBending:
                muscleModeDesc = AutoBendingGenomeDescription().maxAngleDeviation(0.45f).frontBackVelRatio(0.3f);
                break;
            case MuscleMode_ManualBending:
                muscleModeDesc = ManualBendingGenomeDescription().maxAngleDeviation(0.45f).frontBackVelRatio(0.3f);
                break;
            case MuscleMode_AngleBending:
                muscleModeDesc = AngleBendingGenomeDescription().maxAngleDeviation(0.45f).frontBackVelRatio(0.3f);
                break;
            case MuscleMode_AutoCrawling:
                muscleModeDesc = AutoCrawlingGenomeDescription().maxDistanceDeviation(0.45f).frontBackVelRatio(0.3f);
                break;
            case MuscleMode_ManualCrawling:
                muscleModeDesc = ManualCrawlingGenomeDescription().maxDistanceDeviation(0.45f).frontBackVelRatio(0.3f);
                break;
            case MuscleMode_DirectMovement:
                muscleModeDesc = DirectMovementGenomeDescription();
                break;
            default:
                muscleModeDesc = MuscleModeGenomeDescription();
            }
            return MuscleGenomeDescription().mode(muscleModeDesc);
        }
        case CellTypeGenome_Defender:
            return DefenderGenomeDescription().mode(DefenderMode_DefendAgainstInjector);
        case CellTypeGenome_Reconnector:
            return ReconnectorGenomeDescription().restrictToColor(4).restrictToMutants(ReconnectorRestrictToMutants_RestrictToMoreComplexMutants);
        case CellTypeGenome_Detonator:
            return DetonatorGenomeDescription().countdown(23);
        default:
            return CellTypeGenomeDescription_New();
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    DataTransferTests_AllCellTypeGenome_New,
    DataTransferTests_AllCellTypeGenome_New,
    ::testing::Values(
        NodeParameter{CellTypeGenome_Base},
        NodeParameter{CellTypeGenome_Depot},
        NodeParameter{CellTypeGenome_Constructor},
        NodeParameter{CellTypeGenome_Sensor},
        NodeParameter{CellTypeGenome_Oscillator},
        NodeParameter{CellTypeGenome_Attacker},
        NodeParameter{CellTypeGenome_Injector},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_AutoBending},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_ManualBending},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_AngleBending},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_AutoCrawling},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_ManualCrawling},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_DirectMovement},
        NodeParameter{CellTypeGenome_Defender},
        NodeParameter{CellTypeGenome_Reconnector},
        NodeParameter{CellTypeGenome_Detonator}));

TEST_P(DataTransferTests_AllCellTypeGenome_New, singleCell_genome_oneGene_oneNode)
{
    auto cellParameter = GetParam();
    auto cellTypeGenomeDesc = createSomeCellTypeGenomeDescription(cellParameter);

    NeuralNetworkDescription nn1;
    nn1.weight(2, 1, 1.0f);
    NeuralNetworkGenomeDescription nn2;
    nn2.weight(1, 3, -1.0f);

    auto data = CollectionDescription().addCreature(
        CreatureDescription().genes(
            {GeneDescription()
                 .shape(ConstructionShape_Hexagon)
                 .numBranches(3)
                 .numConcatenations(2)
                 .angleAlignment(ConstructorAngleAlignment_180)
                 .stiffness(0.4f)
                 .connectionDistance(0.7f)
                 .nodes({NodeDescription().neuralNetwork(nn2).cellTypeData(cellTypeGenomeDesc).color(3).numRequiredAdditionalConnections(2)})}),
        {CellDescription()
             .neuralNetwork(nn1)
             .id(1)
             .pos({2.0f, 4.0f})
             .vel({0.5f, 1.0f})
             .energy(100.0f)
             .age(1)
             .color(2)
             .barrier(true)
             .cellState(false)
             .creatureId(3534)
             .signalAndRelaxTime({1, 0, -1, 0, 0, 0, 0, 0})});

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests_New, multipleCells_genome_multipleGenes_multiple_Nodes)
{
    auto hexagon = DescriptionEditService::get().createHex(DescriptionEditService::CreateHexParameters().center({100.0f, 100.0f}));
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().genes(
            {GeneDescription().nodes({NodeDescription(), NodeDescription()}),
             GeneDescription().nodes({NodeDescription(), NodeDescription(), NodeDescription()})}),
        hexagon._cells);

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests_New, createCellIds_differentIdsOnDescription)
{
    auto data = CollectionDescription().cells({CellDescription().id(0), CellDescription().id(1)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->addAndSelectSimulationData(std::move(data));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& cell : actualData._cells) {
        ids.insert(cell._id);
    }

    EXPECT_EQ(4, ids.size());
}

TEST_F(DataTransferTests_New, createCellIds_sameIdsOnDescription)
{
    auto data1 = CollectionDescription().cells({CellDescription().id(0), CellDescription().id(0)});
    _simulationFacade->addAndSelectSimulationData(std::move(data1));
    auto data2 = CollectionDescription().cells({CellDescription().id(0), CellDescription().id(0)});
    _simulationFacade->addAndSelectSimulationData(std::move(data2));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& cell : actualData._cells) {
        ids.insert(cell._id);
    }

    EXPECT_EQ(4, ids.size());
}

TEST_F(DataTransferTests_New, createParticleIds_differentIdsOnDescription)
{
    auto data = CollectionDescription().particles({ParticleDescription().id(0), ParticleDescription().id(1)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->addAndSelectSimulationData(std::move(data));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& cell : actualData._particles) {
        ids.insert(cell._id);
    }

    EXPECT_EQ(4, ids.size());
}

TEST_F(DataTransferTests_New, createParticleIds_sameIdsOnDescription)
{
    auto data1 = CollectionDescription().particles({ParticleDescription().id(0), ParticleDescription().id(0)});
    _simulationFacade->addAndSelectSimulationData(std::move(data1));

    auto data2 = CollectionDescription().particles({ParticleDescription().id(0), ParticleDescription().id(0)});
    _simulationFacade->addAndSelectSimulationData(std::move(data2));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& cell : actualData._particles) {
        ids.insert(cell._id);
    }

    EXPECT_EQ(4, ids.size());
}

TEST_F(DataTransferTests_New, createGenomeIds)
{
    CollectionDescription data;
    data.addCreature(CreatureDescription(), {CellDescription()});
    data.addCreature(CreatureDescription(), {CellDescription()});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->addAndSelectSimulationData(std::move(data));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& genome : actualData._creatures) {
        ids.insert(genome._id);
    }

    EXPECT_EQ(4, ids.size());
}

TEST_F(DataTransferTests_New, keepObjectIdsStable)
{
    auto data =
        CollectionDescription().cells({CellDescription().id(0), CellDescription().id(1)}).particles({ParticleDescription().id(2), ParticleDescription().id(3)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->setSimulationData(data);

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> expectedCellIds;
    for (auto const& cell : data._cells) {
        expectedCellIds.insert(cell._id);
    }
    std::unordered_set<uint64_t> actualCellIds;
    for (auto const& cell : actualData._cells) {
        actualCellIds.insert(cell._id);
    }
    EXPECT_EQ(expectedCellIds, actualCellIds);

    std::unordered_set<uint64_t> expectedParticleIds;
    for (auto const& particle : data._particles) {
        expectedParticleIds.insert(particle._id);
    }
    std::unordered_set<uint64_t> actualParticleIds;
    for (auto const& particle : actualData._particles) {
        actualParticleIds.insert(particle._id);
    }
    EXPECT_EQ(expectedParticleIds, actualParticleIds);
}

TEST_F(DataTransferTests_New, createCreatureIds_differentIds)
{
    auto data = CollectionDescription()
                    .cells({
                        CellDescription().creatureId(0),
                        CellDescription().creatureId(1),
                    })
                    .creatures({
                        CreatureDescription().id(0),
                        CreatureDescription().id(1),
                    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->addAndSelectSimulationData(std::move(data));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& creature : actualData._creatures) {
        ids.insert(creature._id);
    }

    EXPECT_EQ(4, ids.size());
}

TEST_F(DataTransferTests_New, createCreatureIds_sameIds)
{
    auto data = CollectionDescription()
                    .cells({
                        CellDescription().creatureId(0),
                        CellDescription().creatureId(0),
                    })
                    .creatures({
                        CreatureDescription().id(0),
                    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->addAndSelectSimulationData(std::move(data));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& creature : actualData._creatures) {
        ids.insert(creature._id);
    }

    EXPECT_EQ(2, ids.size());
}

TEST_F(DataTransferTests_New, changeGenome_successful)
{
    auto data = CollectionDescription().addCreature(CreatureDescription(), {CellDescription()});
    auto creatureId = data._cells.at(0)._creatureId;

    _simulationFacade->setSimulationData(data);

    auto newCreature = CreatureDescription().id(creatureId.value()).genes({GeneDescription().nodes({NodeDescription(), NodeDescription()})});
    auto result = _simulationFacade->changeCreature(newCreature);
    ASSERT_TRUE(result);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto cell = actualData._cells.front();
    EXPECT_EQ(creatureId, cell._creatureId);

    auto creature = actualData._creatures.front();
    EXPECT_EQ(cell._creatureId, creature._id);

    ASSERT_EQ(1, creature._genes.size());

    auto gene = creature._genes.front();
    EXPECT_EQ(2, gene._nodes.size());
}

TEST_F(DataTransferTests_New, changeGenome_failed)
{
    auto data = CollectionDescription().addCreature(CreatureDescription(), {CellDescription()});
    auto creatureId = data._cells.at(0)._creatureId;

    _simulationFacade->setSimulationData(data);

    auto newGenome = CreatureDescription().genes({GeneDescription().nodes({NodeDescription(), NodeDescription()})});
    auto result = _simulationFacade->changeCreature(newGenome);
    ASSERT_FALSE(result);
}

TEST_F(DataTransferTests_New, getInspectedSimulationData)
{
    auto data =
        CollectionDescription()
            .addCreature(
                CreatureDescription().genes({GeneDescription().nodes({NodeDescription(), NodeDescription()})}), {CellDescription(), CellDescription()})
            .addCreature(CreatureDescription(), {CellDescription()});
    auto cellId1 = data._cells.at(0)._id;
    auto cellId2 = data._cells.at(1)._id;

    _simulationFacade->setSimulationData(data);

    auto inspectedData = _simulationFacade->getInspectedSimulationData({cellId1, cellId2});
    ASSERT_EQ(2, inspectedData._cells.size());
    ASSERT_EQ(1, inspectedData._creatures.size());

    auto genome = inspectedData._creatures.front();
    auto cell1 = getCell(inspectedData, cellId1);
    EXPECT_EQ(genome._id, cell1._creatureId);

    auto cell2 = getCell(inspectedData, cellId2);
    EXPECT_EQ(genome._id, cell2._creatureId);

    ASSERT_EQ(1, genome._genes.size());

    auto gene = genome._genes.front();
    EXPECT_EQ(2, gene._nodes.size());
}
