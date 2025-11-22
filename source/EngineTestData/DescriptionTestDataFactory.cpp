#include "DescriptionTestDataFactory.h"

#include <algorithm>

#include "TestHelper.h"

CellDescription DescriptionTestDataFactory::createNonDefaultCellDescription(CellParameter cellParameter) const
{
    CellDescription defaultCell;

    auto cellTypeDesc = createNonDefaultCellTypeDescription(cellParameter);
    auto result = CellDescription()
                      .pos({0.5f, 0.8f})
                      .vel({-0.3f, 0.7f})
                      .energy(150.0f)
                      .age(42)
                      .color(3)
                      .fixed(true)
                      .cellState(false)
                      .geneIndex(42)
                      .nodeIndex(13)
                      .frontAngleId(13)
                      .headCell(true)
                      .parentNodeIndex(14)
                      .signalAndState({1, 0, 0.6f, 0, 0, 0, 0, 0})
                      .signalRestriction(SignalRestrictionDescription().active(true).baseAngle(45.0f).openingAngle(120.0f))
                      .cellType(cellTypeDesc);

    if (cellParameter.cellType != CellType_Structure && cellParameter.cellType != CellType_Free) {
        NeuralNetworkDescription defaultNn;
        NeuralNetworkDescription nn;
        nn.weight(2, 1, 0.7f);
        nn._biases.at(1) = -0.4f;
        nn._activationFunctions.at(5) = 2 % ActivationFunction_Count;
        result._neuralNetwork = nn;
    }
    return result;
}

ParticleDescription DescriptionTestDataFactory::createNonDefaultParticleDescription() const
{
    ParticleDescription defaultParticle;
    return ParticleDescription().id(1).pos({0.3f, 0.9f}).vel({-0.6f, 0.2f}).energy(75.0f).color(5);
}

NodeDescription DescriptionTestDataFactory::createNonDefaultNodeDescription(NodeParameter nodeParameter) const
{
    NodeDescription defaultNode;

    NeuralNetworkGenomeDescription nn;
    nn.weight(4, 3, 0.8f);
    nn._biases.at(3) = -0.5f;
    nn._activationFunctions.at(2) = 1;

    return NodeDescription()
        .neuralNetwork(nn)
        .cellType(createNonDefaultCellTypeGenomeDescription(nodeParameter))
        .color(4)
        .numAdditionalConnections(3)
        .referenceAngle(90.0f)
        .signalRestriction(SignalRestrictionGenomeDescription().active(true).baseAngle(60.0f).openingAngle(180.0f));
}

std::pair<CreatureDescription, GenomeDescription> DescriptionTestDataFactory::createNonDefaultCreatureDescription(NodeParameter nodeParameter) const
{
    CreatureDescription defaultCreature;
    GeneDescription defaultGene;

    auto genome = GenomeDescription()
                      .name("Test Genome")
                      .frontAngle(270.0f)
                      .genes({
                          GeneDescription()
                              .name("Test Gene")
                              .shape(ConstructorShape_Hexagon)
                              .numBranches(4)
                              .separation(true)
                              .numConcatenations(6)
                              .angleAlignment(ConstructorAngleAlignment_180)
                              .stiffness(0.75f)
                              .connectionDistance(0.8f)
                              .nodes({
                                  createNonDefaultNodeDescription(nodeParameter),
                              }),
                      });

    auto creature = CreatureDescription().ancestorId(1001).lineageId(502).generation(7).numCells(25).frontAngleId(42).genomeId(genome._id);

    return {creature, genome};
}

bool DescriptionTestDataFactory::compare(Description left, Description right) const
{
    return TestHelper::compare(left, right);
}

bool DescriptionTestDataFactory::compare(CellDescription left, CellDescription right) const
{
    return TestHelper::compare(left, right);
}

bool DescriptionTestDataFactory::compare(ParticleDescription left, ParticleDescription right) const
{
    return TestHelper::compare(left, right);
}

CellTypeDescription DescriptionTestDataFactory::createNonDefaultCellTypeDescription(CellParameter cellParameter) const
{
    auto const& type = cellParameter.cellType;
    auto muscleMode = cellParameter.muscleMode.value_or(MuscleMode_AutoBending);
    auto sensorMode = cellParameter.sensorMode.value_or(SensorMode_DetectEnergy);

    switch (type) {
    case CellType_Structure:
        return StructureCellDescription();
    case CellType_Free:
        return FreeCellDescription();
    case CellType_Base:
        return BaseDescription();
    case CellType_Depot:
        return DepotDescription();
    case CellType_Constructor: {
        return ConstructorDescription()
            .autoTriggerInterval(50)
            .constructionActivationTime(75)
            .constructionAngle(42.0f)
            .provideEnergy(ProvideEnergy_FreeGeneration)
            .geneIndex(2)
            .lastConstructedCellId(123)
            .currentNodeIndex(1)
            .currentBranch(3)
            .currentConcatenation(2);
    }
    case CellType_Sensor: {
        SensorModeDescription sensorModeDesc;
        switch (sensorMode) {
        case SensorMode_DetectEnergy:
            sensorModeDesc = DetectEnergyDescription().minDensity(0.3f);
            break;
        case SensorMode_DetectStructure:
            sensorModeDesc = DetectStructureDescription();
            break;
        case SensorMode_DetectFreeCell:
            sensorModeDesc = DetectFreeCellDescription().minDensity(0.25f).restrictToColor(2);
            break;
        case SensorMode_DetectCreature:
            sensorModeDesc = DetectCreatureDescription()
                .minNumCells(5)
                .maxNumCells(20)
                .restrictToColor(3)
                .restrictToLineage(DetectCreatureLineageRestriction_SameLineage)
                .lastMatchPos(RealVector2D{10.5f, 20.3f});
            break;
        default:
            sensorModeDesc = SensorModeDescription();
            break;
        }
        return SensorDescription().autoTriggerInterval(80).mode(sensorModeDesc).minRange(10).maxRange(50);
    }
    case CellType_Generator: {
        return GeneratorDescription().autoTriggerInterval(60).alternationInterval(3).numPulses(5);
    }
    case CellType_Attacker:
        return AttackerDescription();
    case CellType_Injector:
        return InjectorDescription().counter(15);
    case CellType_Muscle: {
        MuscleModeDescription muscleModeDesc;
        switch (muscleMode) {
        case MuscleMode_AutoBending: {
            muscleModeDesc = AutoBendingDescription()
                                 .maxAngleDeviation(0.6f)
                                 .forwardBackwardRatio(0.4f)
                                 .initialAngle(135.0f)
                                 .forward(false)
                                 .activation(0.7f)
                                 .activationCountdown(8)
                                 .impulseAlreadyApplied(true);
        } break;
        case MuscleMode_ManualBending:
            muscleModeDesc =
                ManualBendingDescription().maxAngleDeviation(0.5f).forwardBackwardRatio(0.3f).initialAngle(225.0f).lastAngleDelta(0.8f).impulseAlreadyApplied(
                    true);
            break;
        case MuscleMode_AngleBending:
            muscleModeDesc = AngleBendingDescription().maxAngleDeviation(0.7f).attractionRepulsionRatio(0.6f).initialAngle(315.0f);
            break;
        case MuscleMode_AutoCrawling: {
            AutoCrawlingDescription defaultCrawling;
            muscleModeDesc = AutoCrawlingDescription()
                                 .maxDistanceDeviation(0.9f)
                                 .forwardBackwardRatio(0.35f)
                                 .initialDistance(0.6f)
                                 .lastActualDistance(0.8f)
                                 .forward(false)
                                 .activation(0.4f)
                                 .activationCountdown(12)
                                 .impulseAlreadyApplied(true);
        } break;
        case MuscleMode_ManualCrawling:
            muscleModeDesc = ManualCrawlingDescription()
                                 .maxDistanceDeviation(0.75f)
                                 .forwardBackwardRatio(0.45f)
                                 .initialDistance(0.4f)
                                 .lastActualDistance(0.9f)
                                 .lastDistanceDelta(0.65f)
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
        return ReconnectorDescription().restrictToColor(1).restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToMoreComplexMutants);
    case CellType_Detonator:
        return DetonatorDescription().countdown(23);
    default:
        return CellTypeDescription();
    }
}

CellTypeGenomeDescription DescriptionTestDataFactory::createNonDefaultCellTypeGenomeDescription(NodeParameter cellParameter) const
{
    auto const& type = cellParameter.cellTypeGenome;
    auto muscleMode = cellParameter.muscleMode.value_or(MuscleMode_AutoBending);
    auto sensorMode = cellParameter.sensorMode.value_or(SensorMode_DetectEnergy);
    switch (type) {
    case CellTypeGenome_Base:
        return BaseGenomeDescription();
    case CellTypeGenome_Depot:
        return DepotGenomeDescription();
    case CellTypeGenome_Constructor:
        return ConstructorGenomeDescription()
            .autoTriggerInterval(45)
            .constructionActivationTime(85)
            .provideEnergy(ProvideEnergy_FreeGeneration)
            .constructionAngle(30.0f);
    case CellTypeGenome_Sensor: {
        SensorModeGenomeDescription sensorModeDesc;
        switch (sensorMode) {
        case SensorMode_DetectEnergy:
            sensorModeDesc = DetectEnergyGenomeDescription().minDensity(0.25f);
            break;
        case SensorMode_DetectStructure:
            sensorModeDesc = DetectStructureGenomeDescription();
            break;
        case SensorMode_DetectFreeCell:
            sensorModeDesc = DetectFreeCellGenomeDescription().minDensity(0.20f).restrictToColor(6);
            break;
        case SensorMode_DetectCreature:
            sensorModeDesc = DetectCreatureGenomeDescription()
                .minNumCells(3)
                .maxNumCells(15)
                .restrictToColor(4)
                .restrictToLineage(DetectCreatureLineageRestriction_OtherLineage);
            break;
        default:
            sensorModeDesc = SensorModeGenomeDescription();
            break;
        }
        return SensorGenomeDescription().autoTriggerInterval(70).mode(sensorModeDesc).minRange(5).maxRange(30);
    }
    case CellTypeGenome_Generator:
        return GeneratorGenomeDescription().autoTriggerInterval(55).pulseType(GeneratorPulseType_Alternation).alternationInterval(4);
    case CellTypeGenome_Attacker:
        return AttackerGenomeDescription();
    case CellTypeGenome_Injector:
        return InjectorGenomeDescription();
    case CellTypeGenome_Muscle: {
        MuscleModeGenomeDescription muscleModeDesc;
        switch (muscleMode) {
        case MuscleMode_AutoBending:
            muscleModeDesc = AutoBendingGenomeDescription().maxAngleDeviation(0.55f).forwardBackwardRatio(0.25f);
            break;
        case MuscleMode_ManualBending:
            muscleModeDesc = ManualBendingGenomeDescription().maxAngleDeviation(0.45f).forwardBackwardRatio(0.35f);
            break;
        case MuscleMode_AngleBending:
            muscleModeDesc = AngleBendingGenomeDescription().maxAngleDeviation(0.65f).attractionRepulsionRatio(0.55f);
            break;
        case MuscleMode_AutoCrawling:
            muscleModeDesc = AutoCrawlingGenomeDescription().maxDistanceDeviation(0.85f).forwardBackwardRatio(0.15f);
            break;
        case MuscleMode_ManualCrawling:
            muscleModeDesc = ManualCrawlingGenomeDescription().maxDistanceDeviation(0.95f).forwardBackwardRatio(0.25f);
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
        return ReconnectorGenomeDescription().restrictToColor(4).restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToMoreComplexMutants);
    case CellTypeGenome_Detonator: {
        return DetonatorGenomeDescription().countdown(45);
    }
    default:
        return CellTypeGenomeDescription();
    }
}
