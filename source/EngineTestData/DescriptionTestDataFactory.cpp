#include "DescriptionTestDataFactory.h"

#include <algorithm>

#include "EngineInterface/NumberGenerator.h"

CellDescription DescriptionTestDataFactory::createRandomCellDescription(CellParameter cellParameter) const
{
    CellDescription defaultCell;

    auto cellTypeDesc = createRandomCellTypeDescription(cellParameter);
    auto result = CellDescription()
                      .id(1)
                      .pos(getRandomFloat2(0, 1))
                      .vel(getRandomFloat2(-1, 1))
                      .energy(getRandomFloat(50.0f, 200.0f))
                      .age(getRandomInt(defaultCell._age))
                      .color(getRandomInt(defaultCell._color, 0, MAX_COLORS - 1))
                      .barrier(true)
                      .cellState(false)
                      .signalAndRelaxTime({1, 0, getRandomFloat(), 0, 0, 0, 0, 0})
                      .signalRoutingRestriction(
                          SignalRoutingRestrictionDescription().active(true).baseAngle(getRandomFloat(0, 360.0f)).openingAngle(getRandomFloat(0, 360.0f)))
                      .cellTypeData(cellTypeDesc)
                      .metadata(CellMetadataDescription().name("Test1").description("Test2"));

    if (cellParameter.cellType != CellType_Structure && cellParameter.cellType != CellType_Free) {
        NeuralNetworkDescription defaultNn; 
        NeuralNetworkDescription nn;
        nn.weight(2, 1, getRandomFloat());
        nn._biases.at(1) = getRandomFloat();
        nn._activationFunctions.at(5) = getRandomInt(defaultNn._activationFunctions.at(0)) % ActivationFunction_Count;
        result._neuralNetwork = nn;
    }
    return result;
}

ParticleDescription DescriptionTestDataFactory::createRandomParticleDescription() const
{
    ParticleDescription defaultParticle;
    return ParticleDescription()
        .id(1)
        .pos(getRandomFloat2(0, 1))
        .vel(getRandomFloat2(-1, 1))
        .energy(getRandomFloat(50.0f, 200.0f))
        .color(getRandomInt(defaultParticle._color, 0, MAX_COLORS - 1));
}

NodeDescription DescriptionTestDataFactory::createRandomNodeDescription(NodeParameter nodeParameter) const
{
    NodeDescription defaultNode;

    NeuralNetworkGenomeDescription nn;
    nn.weight(4, 3, getRandomFloat());
    nn._biases.at(3) = getRandomFloat();
    nn._activationFunctions.at(2) = getRandomInt(defaultNode._neuralNetwork._activationFunctions.at(0) , 0, ActivationFunction_Count - 1);

    return NodeDescription()
        .neuralNetwork(nn)
        .cellTypeData(createRandomCellTypeGenomeDescription(nodeParameter))
        .color(getRandomInt(defaultNode._color, 0, MAX_COLORS - 1))
        .numAdditionalConnections(getRandomInt())
        .referenceAngle(getRandomFloat(0, 360.0f))
        .signalRoutingRestriction(
            SignalRoutingRestrictionGenomeDescription().active(true).baseAngle(getRandomFloat(0, 360.0f)).openingAngle(getRandomFloat(0, 360.0f)));
}

CreatureDescription DescriptionTestDataFactory::createRandomCreatureDescription(NodeParameter nodeParameter) const
{
    CreatureDescription defaultCreature;
    GeneDescription defaultGene;

    return CreatureDescription()
        .ancestorId(getRandomInt(defaultCreature._ancestorId))
        .mutationId(getRandomInt(defaultCreature._mutationId))
        .generation(getRandomInt(defaultCreature._generation))
        .genomeComplexity(getRandomInt(defaultCreature._genomeComplexity))
        .genome(GenomeDescription()
                    .frontAngle(getRandomFloat(0, 360.0f))
                    .genes({
                        GeneDescription()
                            .shape(ConstructorShape_Hexagon)
                            .numBranches(getRandomInt(defaultGene._numBranches, 1, 6))
                            .separation(false)
                            .numConcatenations(getRandomInt(defaultGene._numConcatenations, 1, 10))
                            .angleAlignment(ConstructorAngleAlignment_180)
                            .stiffness(getRandomFloat(0, 1))
                            .connectionDistance(getRandomFloat(0.3f, 1.0f))
                            .nodes({
                                createRandomNodeDescription(nodeParameter),
                            }),
                    }));
}

bool DescriptionTestDataFactory::compare(CollectionDescription left, CollectionDescription right) const
{
    std::sort(left._cells.begin(), left._cells.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(right._cells.begin(), right._cells.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(left._particles.begin(), left._particles.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(right._particles.begin(), right._particles.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(left._creatures.begin(), left._creatures.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(right._creatures.begin(), right._creatures.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    for (auto& creature : left._creatures) {
        std::sort(creature._cells.begin(), creature._cells.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    }
    for (auto& creature : right._creatures) {
        std::sort(creature._cells.begin(), creature._cells.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    }
    return left == right;
}

bool DescriptionTestDataFactory::compare(CellDescription left, CellDescription right) const
{
    left._id = 0;
    right._id = 0;
    return left == right;
}

bool DescriptionTestDataFactory::compare(ParticleDescription left, ParticleDescription right) const
{
    left._id = 0;
    right._id = 0;
    return left == right;
}

float DescriptionTestDataFactory::getRandomFloat(float min, float max) const
{
    return NumberGenerator::get().getRandomFloat(min, max);
}

RealVector2D DescriptionTestDataFactory::getRandomFloat2(float min, float max) const
{
    return {getRandomFloat(min, max), getRandomFloat(min, max)};
}

int DescriptionTestDataFactory::getRandomInt(int exception, int min, int max) const
{
    while (true) {
        auto result = NumberGenerator::get().getRandomInt(min, max);
        if (result != exception) {
            return result;
        }
    }
}

CellTypeDescription DescriptionTestDataFactory::createRandomCellTypeDescription(CellParameter cellParameter) const
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
    case CellType_Constructor: {
        ConstructorDescription defaultConstructor;
        return ConstructorDescription()
            .autoTriggerInterval(getRandomInt())
            .constructionActivationTime(getRandomInt(defaultConstructor._constructionActivationTime))
            .geneIndex(getRandomInt(defaultConstructor._geneIndex))
            .lastConstructedCellId(getRandomInt())
            .currentNodeIndex(getRandomInt(defaultConstructor._currentNodeIndex))
            .currentBranch(getRandomInt(defaultConstructor._currentBranch))
            .currentConcatenation(getRandomInt(defaultConstructor._currentConcatenation));
    }
    case CellType_Sensor:
        return SensorDescription()
            .autoTriggerInterval(getRandomInt())
            .restrictToColor(getRandomInt(-1, 0, MAX_COLORS - 1))
            .minRange(getRandomInt())
            .maxRange(getRandomInt())
            .minDensity(getRandomFloat(0, 1))
            .restrictToCreatures(SensorRestrictToCreatures_RestrictToLessComplexMutants);
    case CellType_Generator: {
        GeneratorDescription defaultGenerator;
        return GeneratorDescription()
            .autoTriggerInterval(getRandomInt(defaultGenerator._autoTriggerInterval))
            .alternationInterval(getRandomInt(defaultGenerator._alternationInterval))
            .numPulses(getRandomInt(defaultGenerator._numPulses));
    }
    case CellType_Attacker:
        return AttackerDescription();
    case CellType_Injector:
        return InjectorDescription().counter(getRandomInt());
    case CellType_Muscle: {
        MuscleModeDescription muscleModeDesc;
        switch (muscleMode) {
        case MuscleMode_AutoBending: {
            AutoBendingDescription defaultBending;
            muscleModeDesc = AutoBendingDescription()
                                 .maxAngleDeviation(getRandomFloat(0, 1))
                                 .frontBackVelRatio(getRandomFloat(0, 1))
                                 .initialAngle(getRandomFloat(0, 360.0f))
                                 .lastActualAngle(getRandomFloat(0, 360.0))
                                 .forward(false)
                                 .activation(getRandomFloat(0, 1))
                                 .activationCountdown(getRandomInt(defaultBending._activationCountdown))
                                 .impulseAlreadyApplied(true);
        } break;
        case MuscleMode_ManualBending:
            muscleModeDesc = ManualBendingDescription()
                                 .maxAngleDeviation(getRandomFloat(0, 1))
                                 .frontBackVelRatio(getRandomFloat(0, 1))
                                 .initialAngle(getRandomFloat(0, 360.0f))
                                 .lastActualAngle(getRandomFloat(0, 360.0f))
                                 .lastAngleDelta(getRandomFloat(0, 1))
                                 .impulseAlreadyApplied(true);
            break;
        case MuscleMode_AngleBending:
            muscleModeDesc = AngleBendingDescription()
                                 .maxAngleDeviation(getRandomFloat(0, 1))
                                 .frontBackVelRatio(getRandomFloat(0, 1))
                                 .initialAngle(getRandomFloat(0, 360.0f));
            break;
        case MuscleMode_AutoCrawling: {
            AutoCrawlingDescription defaultCrawling;
            muscleModeDesc = AutoCrawlingDescription()
                                 .maxDistanceDeviation(getRandomFloat(0, 1))
                                 .frontBackVelRatio(getRandomFloat(0, 1))
                                 .initialDistance(getRandomFloat(0, 1))
                                 .lastActualDistance(getRandomFloat(0, 1))
                                 .forward(false)
                                 .activation(getRandomFloat(0, 1))
                                 .activationCountdown(getRandomInt(defaultCrawling._activationCountdown))
                                 .impulseAlreadyApplied(true);
        } break;
        case MuscleMode_ManualCrawling:
            muscleModeDesc = ManualCrawlingDescription()
                                 .maxDistanceDeviation(getRandomFloat(0, 1))
                                 .frontBackVelRatio(getRandomFloat(0, 1))
                                 .initialDistance(getRandomFloat(0, 1))
                                 .lastActualDistance(getRandomFloat(0, 1))
                                 .lastDistanceDelta(getRandomFloat(0, 1))
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
        return ReconnectorDescription()
            .restrictToColor(getRandomInt(-1, 0, MAX_COLORS - 1))
            .restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToMoreComplexMutants);
    case CellType_Detonator:
        return DetonatorDescription().countdown(23);
    default:
        return CellTypeDescription();
    }
}

CellTypeGenomeDescription DescriptionTestDataFactory::createRandomCellTypeGenomeDescription(NodeParameter cellParameter) const
{
    auto const& type = cellParameter.cellTypeGenome;
    auto const& muscleMode = cellParameter.muscleMode;
    switch (type) {
    case CellTypeGenome_Base:
        return BaseGenomeDescription();
    case CellTypeGenome_Depot:
        return DepotGenomeDescription();
    case CellTypeGenome_Constructor: {
        ConstructorGenomeDescription defaultConstructor;
        return ConstructorGenomeDescription()
            .autoTriggerInterval(getRandomInt())
            .constructionActivationTime(getRandomInt(defaultConstructor._constructionActivationTime));
    }
    case CellTypeGenome_Sensor:
        return SensorGenomeDescription()
            .autoTriggerInterval(getRandomInt())
            .restrictToColor(getRandomInt(-1, 0, MAX_COLORS - 1))
            .minRange(getRandomInt())
            .maxRange(getRandomInt())
            .minDensity(getRandomFloat(0, 1))
            .restrictToCreatures(SensorRestrictToCreatures_RestrictToLessComplexMutants);
    case CellTypeGenome_Generator: {
        GeneratorGenomeDescription defaultGenerator;
        return GeneratorGenomeDescription()
            .autoTriggerInterval(getRandomInt(defaultGenerator._autoTriggerInterval))
            .pulseType(GeneratorPulseType_Alternation)
            .alternationInterval(getRandomInt(defaultGenerator._alternationInterval));
    }
    case CellTypeGenome_Attacker:
        return AttackerGenomeDescription();
    case CellTypeGenome_Injector:
        return InjectorGenomeDescription();
    case CellTypeGenome_Muscle: {
        MuscleModeGenomeDescription muscleModeDesc;
        switch (muscleMode) {
        case MuscleMode_AutoBending:
            muscleModeDesc = AutoBendingGenomeDescription().maxAngleDeviation(getRandomFloat(0, 1)).frontBackVelRatio(getRandomFloat(0, 1));
            break;
        case MuscleMode_ManualBending:
            muscleModeDesc = ManualBendingGenomeDescription().maxAngleDeviation(getRandomFloat(0, 1)).frontBackVelRatio(getRandomFloat(0, 1));
            break;
        case MuscleMode_AngleBending:
            muscleModeDesc = AngleBendingGenomeDescription().maxAngleDeviation(getRandomFloat(0, 1)).frontBackVelRatio(getRandomFloat(0, 1));
            break;
        case MuscleMode_AutoCrawling:
            muscleModeDesc = AutoCrawlingGenomeDescription().maxDistanceDeviation(getRandomFloat(0, 1)).frontBackVelRatio(getRandomFloat(0, 1));
            break;
        case MuscleMode_ManualCrawling:
            muscleModeDesc = ManualCrawlingGenomeDescription().maxDistanceDeviation(getRandomFloat(0, 1)).frontBackVelRatio(getRandomFloat(0, 1));
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
        return ReconnectorGenomeDescription()
            .restrictToColor(getRandomInt(-1, 0, MAX_COLORS - 1))
            .restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToMoreComplexMutants);
    case CellTypeGenome_Detonator: {
        DetonatorGenomeDescription defaultDetonator;
        return DetonatorGenomeDescription().countdown(getRandomInt(defaultDetonator._countdown));
    }
    default:
        return CellTypeGenomeDescription();
    }
}
