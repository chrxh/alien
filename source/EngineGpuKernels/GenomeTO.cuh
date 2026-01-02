#pragma once

#include <cstdint>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/SimulationParametersTypes.h>

struct NeuralNetworkGenomeTO
{
    float weights[MAX_CHANNELS * MAX_CHANNELS];
    float biases[MAX_CHANNELS];
    ActivationFunction activationFunctions[MAX_CHANNELS];
};

struct BaseGenomeTO
{};

struct DepotGenomeTO
{
    float storageLimit;
    float initialStoredUsableEnergy;
};

struct TelemetryGenomeTO
{};

struct DetectEnergyGenomeTO
{
    float minDensity;
};

struct DetectStructureGenomeTO
{};

struct DetectFreeCellGenomeTO
{
    float minDensity;
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
};

struct DetectCreatureGenomeTO
{
    uint32_t minNumCells;  // 0 = no restriction
    uint32_t maxNumCells;  // 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    LineageRestriction restrictToLineage;
};

union SensorModeGenomeTO
{
    TelemetryGenomeTO telemetry;
    DetectEnergyGenomeTO detectEnergy;
    DetectStructureGenomeTO detectStructure;
    DetectFreeCellGenomeTO detectFreeCell;
    DetectCreatureGenomeTO detectCreature;
};

struct SensorGenomeTO
{
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    SensorMode mode;
    SensorModeGenomeTO modeData;
    uint16_t minRange;
    uint16_t maxRange;
};

struct ConstructorGenomeTO
{
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    uint16_t geneIndex;
    uint16_t constructionActivationTime;
    float constructionAngle;
    ProvideEnergy provideEnergy;
};

struct GeneratorGenomeTO
{
    uint32_t autoTriggerInterval;
    GeneratorPulseType pulseType;
    uint32_t alternationInterval;  // Only for alternation type: 1 = alternate after each pulse, 2 = alternate after second pulse, etc.
};

struct AttackFreeCellGenomeTO
{
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
};

struct AttackCreatureGenomeTO
{
    uint32_t minNumCells;  // 0 = no restriction
    uint32_t maxNumCells;  // 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    LineageRestriction restrictToLineage;
};

union AttackerModeGenomeTO
{
    AttackFreeCellGenomeTO attackFreeCell;
    AttackCreatureGenomeTO attackCreature;
};

struct AttackerGenomeTO
{
    AttackerMode mode;
    AttackerModeGenomeTO modeData;
};

struct InjectorGenomeTO
{
    uint16_t geneIndex;
};

struct AutoBendingGenomeTO
{
    float maxAngleDeviation;     // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1
};

struct ManualBendingGenomeTO
{
    float maxAngleDeviation;     // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1
};

struct AngleBendingGenomeTO
{
    float maxAngleDeviation;         // Between 0 and 1
    float attractionRepulsionRatio;  // Between 0 and 1
};

struct AutoCrawlingGenomeTO
{
    float maxDistanceDeviation;  // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1
};

struct ManualCrawlingGenomeTO
{
    float maxDistanceDeviation;  // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1
};

struct DirectMovementGenomeTO
{};

union MuscleModeGenomeTO
{
    AutoBendingGenomeTO autoBending;
    ManualBendingGenomeTO manualBending;
    AngleBendingGenomeTO angleBending;
    AutoCrawlingGenomeTO autoCrawling;
    ManualCrawlingGenomeTO manualCrawling;
    DirectMovementGenomeTO directMovement;
};

struct MuscleGenomeTO
{
    MuscleMode mode;
    MuscleModeGenomeTO modeData;
};

struct DefenderGenomeTO
{
    DefenderMode mode;
};

struct ReconnectStructureGenomeTO
{};

struct ReconnectFreeCellGenomeTO
{
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
};

struct ReconnectCreatureGenomeTO
{
    uint32_t minNumCells;  // 0 = no restriction
    uint32_t maxNumCells;  // 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    LineageRestriction restrictToLineage;
};

union ReconnectorModeGenomeTO
{
    ReconnectStructureGenomeTO reconnectStructure;
    ReconnectFreeCellGenomeTO reconnectFreeCell;
    ReconnectCreatureGenomeTO reconnectCreature;
};

struct ReconnectorGenomeTO
{
    ReconnectorMode mode;
    ReconnectorModeGenomeTO modeData;
};

struct DetonatorGenomeTO
{
    int32_t countdown;
};

struct DigestorGenomeTO
{
    float rawEnergyConductivity;  // Between 0 and 1
};

struct SignalDelayGenomeTO
{
    uint8_t delay;
};

struct SignalRecorderGenomeTO
{
    bool readOnly;
    uint8_t numWrittenSignalEntries;
};

struct SignalStorageGenomeTO
{
    bool readOnly;
};

struct SignalIntegratorGenomeTO
{
    float newSignalWeight;  // Between 0 and 1
};

union MemoryModeDataGenomeTO
{
    SignalDelayGenomeTO signalDelay;
    SignalRecorderGenomeTO signalRecorder;
    SignalStorageGenomeTO signalStorage;
    SignalIntegratorGenomeTO signalIntegrator;
};

struct SignalEntryGenomeTO
{
    float channels[MAX_CHANNELS];
};

struct MemoryGenomeTO
{
    MemoryMode mode;
    MemoryModeDataGenomeTO modeData;

    uint8_t numSignalEntries;
    uint64_t signalEntriesDataIndex;  // Heap index to SignalEntryGenomeTO[MAX_CELL_MEMORY_ENTRIES]
};

union CellTypeDataGenomeTO
{
    BaseGenomeTO base;
    DepotGenomeTO depot;
    ConstructorGenomeTO constructor;
    SensorGenomeTO sensor;
    GeneratorGenomeTO generator;
    AttackerGenomeTO attacker;
    InjectorGenomeTO injector;
    MuscleGenomeTO muscle;
    DefenderGenomeTO defender;
    ReconnectorGenomeTO reconnector;
    DetonatorGenomeTO detonator;
    DigestorGenomeTO digestor;
    MemoryGenomeTO memory;
};

struct SignalRestrictionGenomeTO
{
    SignalRestrictionMode mode;
    float baseAngle;
    float openingAngle;
};


struct NodeTO
{
    float referenceAngle;
    int color;
    int numAdditionalConnections;

    NeuralNetworkGenomeTO neuralNetwork;
    CellTypeGenome cellType;
    CellTypeDataGenomeTO cellTypeData;
    SignalRestrictionGenomeTO signalRestriction;
};

struct GeneTO
{
    Char64 name;
    ConstructorShape shape;
    bool separation;
    uint8_t numBranches;  // For separation = false
    ConstructorAngleAlignment angleAlignment;
    float stiffness;
    float connectionDistance;
    int numConcatenations;

    int numNodes;
    uint64_t nodeArrayIndex;
};

struct GenomeTO
{
    uint64_t id;
    Char64 name;
    int numGenes;
    uint64_t geneArrayIndex;

    float frontAngle;

    // Temporary data
    uint64_t genomeIndexOnGpu;
};
