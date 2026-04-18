#pragma once

#include <cstdint>

#include <cuda_fp16.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/NeuralNetWeight.h>
#include <EngineInterface/SimulationParametersTypes.h>

struct NeuralNetGenome
{
    NeuralNetWeight weights[NEURONS_PER_CELL * NEURONS_PER_CELL];
    float biases[NEURONS_PER_CELL];
    ActivationFunction activationFunctions[NEURONS_PER_CELL];
    float connectionWeights[MAX_OBJECT_CONNECTIONS];
};

struct BaseGenome
{};

struct VoidGenome
{};

struct DepotGenome
{
    float storageLimit;
    float initialStoredUsableEnergy;
};

struct TelemetryGenome
{};

struct DetectEnergyGenome
{
    float minDensity;
};

struct DetectSolidGenome
{};

struct DetectFreeCellGenome
{
    float minDensity;
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
};

struct DetectCreatureGenome
{
    uint32_t minNumCells;      // 0 = no restriction
    uint32_t maxNumCells;      // 0 = no restriction
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
    LineageRestriction restrictToLineage;
};

union SensorModeGenome
{
    TelemetryGenome telemetry;
    DetectEnergyGenome detectEnergy;
    DetectSolidGenome detectSolid;
    DetectFreeCellGenome detectFreeCell;
    DetectCreatureGenome detectCreature;
};

struct SensorGenome
{
    bool autoTrigger;
    bool tagForAttackers;
    SensorMode mode;
    SensorModeGenome modeData;
    uint16_t minRange;
    uint16_t maxRange;
};

struct ConstructorGenome
{
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    uint16_t geneIndex;
    uint16_t constructionActivationTime;
    float constructionAngle;
    ProvideEnergy provideEnergy;
};

struct SquareSignalGenome
{
    float amplitude;
    int period;
};

struct SawtoothSignalGenome
{
    float amplitude;
    int period;
};

union GeneratorModeGenome
{
    SquareSignalGenome squareSignal;
    SawtoothSignalGenome sawtoothSignal;
};

struct GeneratorGenome
{
    bool additive;
    float valueOffset;
    int timeOffset;
    GeneratorMode mode;
    GeneratorModeGenome modeData;
};

struct AttackFreeCellGenome
{
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
};

struct AttackCreatureGenome
{};

union AttackerModeGenome
{
    AttackFreeCellGenome attackFreeCell;
    AttackCreatureGenome attackCreature;
};

struct AttackerGenome
{
    AttackerMode mode;
    AttackerModeGenome modeData;
};

struct InjectorGenome
{
    uint16_t geneIndex;
};

struct AutoBendingGenome
{
    float maxAngleDeviation;     // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1
};

struct ManualBendingGenome
{
    float maxAngleDeviation;     // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1
};

struct AngleBendingGenome
{
    float maxAngleDeviation;         // Between 0 and 1
    float attractionRepulsionRatio;  // Between 0 and 1
};

struct AutoCrawlingGenome
{
    float maxDistanceDeviation;  // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1
};

struct ManualCrawlingGenome
{
    float maxDistanceDeviation;  // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1
};

struct DirectMovementGenome
{};

union MuscleModeGenome
{
    AutoBendingGenome autoBending;
    ManualBendingGenome manualBending;
    AngleBendingGenome angleBending;
    AutoCrawlingGenome autoCrawling;
    ManualCrawlingGenome manualCrawling;
    DirectMovementGenome directMovement;
};

struct MuscleGenome
{
    MuscleMode mode;
    MuscleModeGenome modeData;
};

struct DefenderGenome
{
    DefenderMode mode;
};

struct ReconnectSolidGenome
{};

struct ReconnectFreeCellGenome
{
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
};

struct ReconnectCreatureGenome
{
    uint32_t minNumCells;     // 0 = no restriction
    uint32_t maxNumCells;     // 0 = no restriction
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
    LineageRestriction restrictToLineage;
};

union ReconnectorModeGenome
{
    ReconnectSolidGenome reconnectSolid;
    ReconnectFreeCellGenome reconnectFreeCell;
    ReconnectCreatureGenome reconnectCreature;
};

struct ReconnectorGenome
{
    ReconnectorMode mode;
    ReconnectorModeGenome modeData;
};

struct DetonatorGenome
{
    int32_t countdown;
};

struct DigestorGenome
{
    float rawEnergyConductivity;  // Between 0 and 1
};

struct SignalDelayGenome
{
    uint8_t delay;
};

struct SignalRecorderGenome
{
    bool readOnly;
    uint8_t numWrittenSignalEntries;
};

struct SignalStorageGenome
{
    bool readOnly;
};

struct SignalIntegratorGenome
{
    float newSignalWeight;  // Between 0 and 1
};

union MemoryModeDataGenome
{
    SignalDelayGenome signalDelay;
    SignalRecorderGenome signalRecorder;
    SignalStorageGenome signalStorage;
    SignalIntegratorGenome signalIntegrator;
};

struct SignalEntryGenome
{
    float channels[NEURONS_PER_CELL];
};

struct MemoryGenome
{
    MemoryMode mode;
    MemoryModeDataGenome modeData;

    uint8_t numSignalEntries;
    uint16_t channelBitMask;
    SignalEntryGenome* signalEntries;  // Pointer to heap memory
};

struct SenderGenome
{
    uint8_t range;
    int maxTimesSent;
};

struct ReceiverGenome
{
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
    LineageRestriction restrictToLineage;
};

union CommunicatorModeDataGenome
{
    SenderGenome sender;
    ReceiverGenome receiver;
};

struct CommunicatorGenome
{
    CommunicatorMode mode;
    CommunicatorModeDataGenome modeData;
};

union CellTypeDataGenome
{
    BaseGenome base;
    DepotGenome depot;
    SensorGenome sensor;
    GeneratorGenome generator;
    AttackerGenome attacker;
    InjectorGenome injector;
    MuscleGenome muscle;
    DefenderGenome defender;
    ReconnectorGenome reconnector;
    DetonatorGenome detonator;
    DigestorGenome digestor;
    MemoryGenome memory;
    CommunicatorGenome communicator;
    VoidGenome voidCell;
};

struct Node
{
    float referenceAngle;
    int color;

    NeuralNetGenome neuralNetwork;
    CellType cellType;
    CellTypeDataGenome cellTypeData;
    bool constructorAvailable;      // If true, constructor holds valid data
    ConstructorGenome constructor;  // Optional constructor data
};

struct Gene
{
    Char64 name;
    ConstructorShape shape;
    bool separation;
    uint8_t numBranches;  // For separation = false
    float stiffness;
    float connectionDistance;
    int numConcatenations;

    int numNodes;
    Node* nodes;
};

struct NeuronMutation
{
    float probability;
    float weightSigma;
    float biasSigma;
    float activationFunctionProbability;
};

struct ConnectionMutation
{
    float probability;
    float sigma;
};

struct Genome
{
    uint64_t id;
    Char64 name;
    int numGenes;
    Gene* genes;

    uint32_t lineageId;
    uint32_t prevLineageId;
    float frontAngle;
    float lineageMutationProbability;

    NeuronMutation neuronMutation1;
    NeuronMutation neuronMutation2;
    ConnectionMutation connectionMutationRate1;
    ConnectionMutation connectionMutationRate2;

    // Temporary data
    uint64_t genomeIndex;  // May be invalid

    __device__ __inline__ bool isRelatedLineage(Genome* other)
    {
        if (prevLineageId != 0 && other->prevLineageId != 0) {
            return lineageId == other->lineageId || lineageId == other->prevLineageId || prevLineageId == other->lineageId
                || prevLineageId == other->prevLineageId;
        } else {
            return lineageId == other->lineageId;
        }
    }
};
