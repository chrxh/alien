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
                .constructionActivationTime(4)
                .constructionAngle1(34.4f)
                .constructionAngle2(-45.5f)
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
    auto cellTypeGenomeDesc = createSomeCellTypeDescription(cellParameter);

    CollectionDescription data;
    NeuralNetworkDescription nn;
    nn.weight(2, 1, 1.0f);
    data.addCell(CellDescription()
                     .neuralNetwork(nn)
                     .id(1)
                     .pos({2.0f, 4.0f})
                     .vel({0.5f, 1.0f})
                     .energy(100.0f)
                     .age(1)
                     .color(2)
                     .barrier(true)
                     .livingState(false)
                     .creatureId(3534)
                     .signal({1, 0, -1, 0, 0, 0, 0, 0})
                     .signalRoutingRestriction(SignalRoutingRestrictionDescription().active(true).baseAngle(23.0f).openingAngle(42.0f))
                     .cellType(cellTypeGenomeDesc)
                     .metadata(CellMetadataDescription().name("Test1").description("Test2")));

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
            return ConstructorGenomeDescription_New().autoTriggerInterval(7).constructionActivationTime(4).constructionAngle1(34.4f).constructionAngle2(-45.5f);
        case CellTypeGenome_Sensor:
            return SensorGenomeDescription().autoTriggerInterval(3).restrictToColor(5).minRange(34).maxRange(67).minDensity(0.25f).restrictToMutants(
                SensorRestrictToMutants_RestrictToLessComplexMutants);
        case CellTypeGenome_Oscillator:
            return OscillatorGenomeDescription().autoTriggerInterval(27).alternationInterval(45);
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
        GenomeDescription_New().genes({GeneDescription().nodes({NodeDescription().neuralNetwork(nn2).cellTypeData(cellTypeGenomeDesc)})}),
        {CellDescription()
             .neuralNetwork(nn1)
             .id(1)
             .pos({2.0f, 4.0f})
             .vel({0.5f, 1.0f})
             .energy(100.0f)
             .age(1)
             .color(2)
             .barrier(true)
             .livingState(false)
             .creatureId(3534)
             .signal({1, 0, -1, 0, 0, 0, 0, 0})});

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests_New, multipleCells_genome_multipleGenes_multiple_Nodes)
{
    auto hexagon = DescriptionEditService::get().createHex(DescriptionEditService::CreateHexParameters().center({100.0f, 100.0f}));
    CollectionDescription data;
    data.addCreature(
        GenomeDescription_New().genes(
            {GeneDescription().nodes({NodeDescription(), NodeDescription()}),
             GeneDescription().nodes({NodeDescription(), NodeDescription(), NodeDescription()})}),
        hexagon._cells);

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}
