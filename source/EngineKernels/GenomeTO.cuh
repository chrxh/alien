#pragma once

#include <cstdint>

#include <cuda_fp16.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/NeuralNetWeight.h>
#include <EngineInterface/SimulationParametersTypes.h>

struct NeuralNetGenomeTO
{
    NeuralNetWeight weights[NEURONS_PER_CELL * NEURONS_PER_CELL];
    float biases[NEURONS_PER_CELL];
    ActivationFunction activationFunctions[NEURONS_PER_CELL];
    float connectionWeights[MAX_OBJECT_CONNECTIONS];
};

struct BaseGenomeTO
{};

struct VoidGenomeTO
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

struct DetectSolidGenomeTO
{};

struct DetectFreeCellGenomeTO
{
    float minDensity;
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
};

struct DetectCreatureGenomeTO
{
    uint32_t minNumCells;       // 0 = no restriction
    uint32_t maxNumCells;       // 0 = no restriction
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
    LineageRestriction restrictToLineage;
};

union SensorModeGenomeTO
{
    TelemetryGenomeTO telemetry;
    DetectEnergyGenomeTO detectEnergy;
    DetectSolidGenomeTO detectSolid;
    DetectFreeCellGenomeTO detectFreeCell;
    DetectCreatureGenomeTO detectCreature;
};

struct SensorGenomeTO
{
    bool autoTrigger;
    bool tagForAttackers;
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
    float reservedEnergy;
    bool separation;
    uint8_t numBranches;  // For separation = false
    int numConcatenations;
};

struct SquareSignalGenomeTO
{
    int period;
};

struct SawtoothSignalGenomeTO
{
    int period;
};

union GeneratorModeGenomeTO
{
    SquareSignalGenomeTO squareSignal;
    SawtoothSignalGenomeTO sawtoothSignal;
};

struct GeneratorGenomeTO
{
    bool additive;
    float minValue;
    float maxValue;
    int timeOffset;
    GeneratorMode mode;
    GeneratorModeGenomeTO modeData;
};

struct AttackFreeCellGenomeTO
{
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
};

struct AttackCreatureGenomeTO
{
    uint32_t minNumCells;      // 0 = no restriction
    uint32_t maxNumCells;      // 0 = no restriction
    uint8_t restrictToColors;  // 0 ... 6 = color restriction, 255 = no restriction
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

struct ReconnectSolidGenomeTO
{};

struct ReconnectFreeCellGenomeTO
{
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
};

struct ReconnectCreatureGenomeTO
{
    uint32_t minNumCells;       // 0 = no restriction
    uint32_t maxNumCells;       // 0 = no restriction
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
    LineageRestriction restrictToLineage;
};

union ReconnectorModeGenomeTO
{
    ReconnectSolidGenomeTO reconnectSolid;
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
    float channels[NEURONS_PER_CELL];
};

struct MemoryGenomeTO
{
    MemoryMode mode;
    MemoryModeDataGenomeTO modeData;

    uint8_t numSignalEntries;
    uint16_t channelBitMask;
    uint64_t signalEntriesDataIndex;
};

struct SenderGenomeTO
{
    uint8_t range;
    int maxTimesSent;
};

struct ReceiverGenomeTO
{
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
    LineageRestriction restrictToLineage;
};

union CommunicatorModeGenomeTO
{
    SenderGenomeTO sender;
    ReceiverGenomeTO receiver;
};

struct CommunicatorGenomeTO
{
    CommunicatorMode mode;
    CommunicatorModeGenomeTO modeData;
};

union CellTypeDataGenomeTO
{
    BaseGenomeTO base;
    DepotGenomeTO depot;
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
    CommunicatorGenomeTO communicator;
    VoidGenomeTO voidCell;
};

struct NodeTO
{
    float referenceAngle;
    int color;

    NeuralNetGenomeTO neuralNetwork;
    CellType cellType;
    CellTypeDataGenomeTO cellTypeData;
    bool constructorAvailable;        // If true, constructor holds valid data
    ConstructorGenomeTO constructor;  // Optional constructor data
};

struct GeneTO
{
    Char64 name;
    ConstructorShape shape;
    float stiffness;
    float connectionDistance;

    int numNodes;
    uint64_t nodeArrayIndex;
};

struct NeuronMutationTO
{
    float eventProbability;
    float weightSigma;
    float biasSigma;
    float activationFunctionProbability;
};

struct ConnectionMutationTO
{
    float eventProbability;
    float sigma;
};

struct CellTypePropertiesMutationTO
{
    float eventProbability;
    float sigma;
    float probability;
};

struct MutationRatesTO
{
    NeuronMutationTO neuronMutation1;
    NeuronMutationTO neuronMutation2;
    ConnectionMutationTO connectionMutation1;
    ConnectionMutationTO connectionMutation2;
    CellTypePropertiesMutationTO cellTypePropertiesMutation1;
    CellTypePropertiesMutationTO cellTypePropertiesMutation2;
};

struct GenomeTO
{
    uint64_t id;
    Char64 name;
    int numGenes;
    uint64_t geneArrayIndex;

    uint32_t lineageId;
    uint32_t prevLineageId;
    float frontAngle;
    float accumulatedMutations;
    MutationRatesTO mutationRates;

    // Temporary data
    uint64_t genomeIndexOnGpu;
};
