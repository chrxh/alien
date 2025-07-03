#include "DescriptionTestDataFactory.h"

#include <algorithm>

CellDescription DescriptionTestDataFactory::createRandomCellDescription(CellParameter cellParameter) const
{
    auto cellTypeDesc = createRandomCellTypeDescription(cellParameter);
    auto result = CellDescription()
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
        nn._biases.at(1) = -0.2f;
        nn._activationFunctions.at(5) = ActivationFunction_Gaussian;
        result._neuralNetwork = nn;
    }
    return result;
}

ParticleDescription DescriptionTestDataFactory::createRandomParticleDescription() const
{
    return ParticleDescription().id(1).pos({2.0f, 4.0f}).vel({0.5f, 1.0f}).energy(100.0f).color(2);
}

CreatureDescription DescriptionTestDataFactory::createRandomCreatureDescription(NodeParameter nodeParameter) const
{
    NeuralNetworkDescription nn1;
    nn1.weight(2, 1, 1.0f);
    NeuralNetworkGenomeDescription nn2;
    nn2.weight(1, 3, -1.0f);

    return CreatureDescription().ancestorId(123).mutationId(42).genomeComplexity(23).genome(GenomeDescription().frontAngle(34.0f).genes({
        GeneDescription()
            .shape(ConstructionShape_Hexagon)
            .numBranches(3)
            .numConcatenations(2)
            .angleAlignment(ConstructorAngleAlignment_180)
            .stiffness(0.4f)
            .connectionDistance(0.7f)
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
    case CellType_Constructor:
        return ConstructorDescription().autoTriggerInterval(7).geneIndex(3).constructionActivationTime(4).constructionAngle(34.4f).lastConstructedCellId(45ull);
    case CellType_Sensor:
        return SensorDescription().autoTriggerInterval(3).restrictToColor(5).minRange(34).maxRange(67).minDensity(0.25f).restrictToCreatures(
            SensorRestrictToCreatures_RestrictToLessComplexMutants);
    case CellType_Generator:
        return GeneratorDescription().autoTriggerInterval(27).alternationInterval(45).numPulses(23);
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
        return ReconnectorDescription().restrictToColor(4).restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToMoreComplexMutants);
    case CellType_Detonator:
        return DetonatorDescription().countdown(23);
    default:
        return CellTypeDescription();
    }
}

NodeDescription DescriptionTestDataFactory::createRandomNodeDescription(NodeParameter nodeParameter) const
{
    NeuralNetworkGenomeDescription nn;
    nn.weight(4, 3, -0.7f);
    nn._biases.at(3) = 0.3f;
    nn._activationFunctions.at(2) = ActivationFunction_Gaussian;

    return NodeDescription()
        .neuralNetwork(nn)
        .cellTypeData(createRandomCellTypeGenomeDescription(nodeParameter))
        .color(3)
        .numRequiredAdditionalConnections(2)
        .referenceAngle(56.0f)
        .signalRoutingRestriction(SignalRoutingRestrictionGenomeDescription().active(true).baseAngle(34.0f).openingAngle(67.0f));
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
    case CellTypeGenome_Constructor:
        return ConstructorGenomeDescription().autoTriggerInterval(7).constructionActivationTime(4).constructionAngle(34.4f);
    case CellTypeGenome_Sensor:
        return SensorGenomeDescription().autoTriggerInterval(3).restrictToColor(5).minRange(34).maxRange(67).minDensity(0.25f).restrictToCreatures(
            SensorRestrictToCreatures_RestrictToLessComplexMutants);
    case CellTypeGenome_Generator:
        return GeneratorGenomeDescription().autoTriggerInterval(27).pulseType(GeneratorPulseType_Alternation).alternationInterval(45);
    case CellTypeGenome_Attacker:
        return AttackerGenomeDescription();
    case CellTypeGenome_Injector:
        return InjectorGenomeDescription();
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
        return ReconnectorGenomeDescription().restrictToColor(4).restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToMoreComplexMutants);
    case CellTypeGenome_Detonator:
        return DetonatorGenomeDescription().countdown(23);
    default:
        return CellTypeGenomeDescription();
    }
}
