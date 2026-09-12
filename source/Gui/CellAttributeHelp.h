#pragma once

#include <string>

// Identifies an attribute of an object, a cell, a genome, a gene or a genome node that is shown in one of the editors or in the inspection window
enum class CellAttribute
{
    // Object
    ObjectId,
    Position,
    Velocity,
    Stiffness,
    Color,
    Static,
    Sticky,
    ObjectType,
    ConnectedId,
    ConnectionDistance,
    ConnectionRefAngle,

    // Energy particle
    ParticleId,
    ParticleEnergy,

    // Solid, fluid and free cell
    SolidEnergy,
    FluidEnergy,
    FluidGlow,
    FreeCellEnergy,
    FreeCellAge,

    // Cell
    CellUsableEnergy,
    CellRawEnergy,
    CellFrontAngle,
    CellAge,
    CellState,
    CellNodeIndex,
    CellParentNodeIndex,
    CellGeneIndex,
    CellConcatenationIndex,
    CellBranchIndex,
    CellActivationTime,
    CellHeadCell,
    CellType,
    CellHasConstructor,

    // Genome
    GenomeName,
    GenomeFrontAngle,
    GenomeResistanceToInjection,
    GenomeApplyMetaMutations,
    GenomeMutationRatesEdit,

    // Gene
    GeneName,
    GeneShape,
    GeneConnectionDistance,
    GeneStiffness,
    GeneHomogeneousCellType,

    // Mutation rates
    MutationConnectionProbability,
    MutationConnectionValueChangeSigma,
    MutationNeuronProbability,
    MutationNeuronWeightChangeSigma,
    MutationNeuronBiasChangeSigma,
    MutationNeuronActfnChangeProbability,
    MutationCellTypePropertiesProbability,
    MutationCellTypePropertiesValueChangeSigma,
    MutationCellTypePropertiesEnumChangeProbability,
    MutationGeometryProbability,
    MutationGeometryValueChangeSigma,
    MutationGeometryEnumChangeProbability,
    MutationCellTypeModeProbability,
    MutationCellTypeProbability,
    MutationCustomizationProbability,
    MutationVoidProbability,
    MutationExtendGeneProbability,
    MutationAddNodeProbability,
    MutationTrimGeneProbability,
    MutationDeleteNodeProbability,
    MutationDuplicateGeneProbability,
    MutationDeleteGeneProbability,
    MutationCopyNodeSectionProbability,
    MutationMoveNodeSectionProbability,
    MutationConstructorProbability,
    MutationConstructorValueChangeSigma,
    MutationConstructorEnumChangeProbability,
    MutationConstructorToggleProbability,

    // Genome node
    NodeAngle,
    NodeCustomization,
    NodeConstructionGene,

    // Constructor
    ConstructorAutoTriggerInterval,
    ConstructorActivationTime,
    ConstructorConstructionAngle,
    ConstructorProvideEnergy,
    ConstructorReservedEnergy,
    ConstructorSeparation,
    ConstructorNumBranches,
    ConstructorNumConcatenations,
    ConstructorGeneIndex,

    // Depot
    DepotStorageLimit,
    DepotInitialStoredEnergy,
    DepotStoredEnergy,

    // Sensor
    SensorAutoTrigger,
    SensorTagForAttackers,
    SensorMode,
    SensorEnergyMinDensity,
    SensorFreeCellMinDensity,
    SensorMinRange,
    SensorMaxRange,
    SensorRestrictToColors,
    SensorRestrictToLineage,
    SensorMinNumCells,
    SensorMaxNumCells,

    // Generator
    GeneratorAdditive,
    GeneratorMinValue,
    GeneratorMaxValue,
    GeneratorTimeOffset,
    GeneratorMode,
    GeneratorPeriod,

    // Attacker
    AttackerMode,
    AttackerRestrictToColors,

    // Injector
    InjectorGeneIndex,

    // Muscle
    MuscleMode,
    MuscleMaxAngleDeviation,
    MuscleForwardBackwardRatio,
    MuscleAttractionRepulsionRatio,
    MuscleMaxDistanceDeviation,

    // Defender
    DefenderMode,

    // Reconnector
    ReconnectorMode,
    ReconnectorRestrictToColors,
    ReconnectorRestrictToLineage,
    ReconnectorMinNumCells,
    ReconnectorMaxNumCells,

    // Detonator
    DetonatorState,
    DetonatorCountdown,

    // Digestor
    DigestorEnergyConductivity,
    DigestorEnergyConversion,

    // Memory
    MemoryMode,
    MemoryDelay,
    MemoryReadOnly,
    MemoryNewSignalWeight,
    MemoryChannelMask,
    MemorySignalBuffer,

    // Communicator
    CommunicatorMode,
    CommunicatorRange,
    CommunicatorOneway,
    CommunicatorRestrictToColors,
    CommunicatorRestrictToLineage,

    // Neural activity
    NeuralSignal,
    NeuralMemory,

    // Creature
    CreatureId,
    CreatureGeneration,
    CreatureNumCells,
    CreatureLineageId,
    CreatureMutationsInLineage,
    CreatureMutationsTotal,
    CreatureGenomeName,
    CreatureResistanceToInjection,
    CreatureApplyMetaMutations,
    CreatureEditGenome,

    Count
};

class CellAttributeHelp
{
public:
    static std::string get(CellAttribute attribute);
};
