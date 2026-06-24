#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <EngineInterface/ArraySizesForTOs.h>
#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>

#include "GenomeTO.cuh"
#include <stdint.h>

struct EnergyTO
{
    uint64_t id;
    float energy;
    float2 pos;
    float2 vel;
    uint8_t color;

    uint8_t selected;
};

struct ConnectionTO
{
    uint64_t objectIndex;  // May be invalid

    float distance;
    float angleFromPrevious;
};

struct NeuralNetTO
{
    NeuralNetWeight weights[NEURONS_PER_CELL * NEURONS_PER_CELL];
    float biases[NEURONS_PER_CELL];
    ActivationFunction activationFunctions[NEURONS_PER_CELL];
    float connectionWeights[MAX_OBJECT_CONNECTIONS];
};

struct BaseTO
{};

struct VoidTO
{};

struct DepotTO
{
    float storageLimit;
    float storedUsableEnergy;
};

struct ConstructorTO
{
    // Properties
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    uint16_t constructionActivationTime;
    float constructionAngle;
    ProvideEnergy provideEnergy;
    float reservedEnergy;
    bool separation;
    uint8_t numBranches;  // For separation = false
    int numConcatenations;

    // Genome data
    uint16_t geneIndex;

    // Process data
    uint64_t lastConstructedCellId;  // May be invalid
    uint16_t currentOffspring;
};

struct TelemetryTO
{};

struct DetectEnergyTO
{
    float minDensity;
};

struct DetectSolidTO
{};

struct DetectFreeCellTO
{
    float minDensity;
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
};

struct DetectCreatureTO
{
    uint32_t minNumCells;       // 0 = no restriction
    uint32_t maxNumCells;       // 0 = no restriction
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
    LineageRestriction restrictToLineage;
};

union SensorModeTO
{
    TelemetryTO telemetry;
    DetectEnergyTO detectEnergy;
    DetectSolidTO detectSolid;
    DetectFreeCellTO detectFreeCell;
    DetectCreatureTO detectCreature;
};

struct SensorLastMatchTO
{
    uint16_t creatureIdPart;
    float2 pos;
};

struct SensorTO
{
    bool autoTrigger;
    bool tagForAttackers;
    SensorMode mode;
    SensorModeTO modeData;
    uint16_t minRange;
    uint16_t maxRange;

    // Process data
    bool lastMatchAvailable;
    SensorLastMatchTO lastMatch;
};

struct SquareSignalTO
{
    int period;
};

struct SawtoothSignalTO
{
    int period;
};

union GeneratorModeTO
{
    SquareSignalTO squareSignal;
    SawtoothSignalTO sawtoothSignal;
};

struct GeneratorTO
{
    bool additive;
    float minValue;
    float maxValue;
    int timeOffset;
    GeneratorMode mode;
    GeneratorModeTO modeData;

    // Process data
    uint32_t numPulses;
};

struct AttackFreeCellTO
{
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
};

struct AttackCreatureTO
{};

union AttackerModeTO
{
    AttackFreeCellTO attackFreeCell;
    AttackCreatureTO attackCreature;
};

struct AttackerTO
{
    AttackerMode mode;
    AttackerModeTO modeData;
};

struct InjectorTO
{
    uint16_t geneIndex;
};

struct AutoBendingTO
{
    // Fixed data
    float maxAngleDeviation;     // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1

    // Process data
    float initialAngle;  // May be invalid
    bool forward;        // Current direction
};

struct ManualBendingTO
{
    // Fixed data
    float maxAngleDeviation;     // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1

    // Process data
    float initialAngle;  // May be invalid
    float lastAngleDelta;
};

struct AngleBendingTO
{
    // Fixed data
    float maxAngleDeviation;         // Between 0 and 1
    float attractionRepulsionRatio;  // Between 0 and 1

    // Process data
    float initialAngle;  // May be invalid
};

struct AutoCrawlingTO
{
    // Fixed data
    float maxDistanceDeviation;  // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1

    // Process data
    float initialDistance;  // May be invalid
    float lastActualDistance;
    bool forward;  // Current direction
};


struct ManualCrawlingTO
{
    // Fixed data
    float maxDistanceDeviation;  // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1

    // Process data
    float initialDistance;  // May be invalid
    float lastActualDistance;
    float lastDistanceDelta;
};

struct DirectMovementTO
{};

union MuscleModeTO
{
    AutoBendingTO autoBending;
    ManualBendingTO manualBending;
    AngleBendingTO angleBending;
    AutoCrawlingTO autoCrawling;
    ManualCrawlingTO manualCrawling;
    DirectMovementTO directMovement;
};

struct MuscleTO
{
    MuscleMode mode;
    MuscleModeTO modeData;

    // Additional rendering data
    float lastMovementX;
    float lastMovementY;
};

struct DefenderTO
{
    DefenderMode mode;
};

struct ReconnectSolidTO
{};

struct ReconnectFreeCellTO
{
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
};

struct ReconnectCreatureTO
{
    uint32_t minNumCells;       // 0 = no restriction
    uint32_t maxNumCells;       // 0 = no restriction
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
    LineageRestriction restrictToLineage;
};

union ReconnectorModeTO
{
    ReconnectSolidTO reconnectSolid;
    ReconnectFreeCellTO reconnectFreeCell;
    ReconnectCreatureTO reconnectCreature;
};

struct ReconnectorTO
{
    ReconnectorMode mode;
    ReconnectorModeTO modeData;
};

struct DetonatorTO
{
    DetonatorState state;
    int32_t countdown;
};

struct DigestorTO
{
    float rawEnergyConductivity;  // Between 0 and 1
};

struct SignalDelayTO
{
    uint8_t delay;
    uint8_t numSignalEntriesInitialized;
    uint8_t ringBufferIndex;
};

struct SignalRecorderTO
{
    bool readOnly;
    SignalRecorderState state;
    uint8_t numWrittenSignalEntries;
    uint8_t numReadSignalEntries;
};

struct SignalStorageTO
{
    bool readOnly;
};

struct SignalIntegratorTO
{
    float newSignalWeight;  // Between 0 and 1
};

union MemoryModeDataTO
{
    SignalDelayTO signalDelay;
    SignalRecorderTO signalRecorder;
    SignalStorageTO signalStorage;
    SignalIntegratorTO signalIntegrator;
};

struct SignalEntryTO
{
    float channels[NEURONS_PER_CELL];
};

struct MemoryTO
{
    MemoryMode mode;
    MemoryModeDataTO modeData;

    uint8_t numSignalEntries;
    uint16_t channelBitMask;
    uint64_t signalEntriesDataIndex;  // Heap index to SignalEntryTO[MAX_CELL_MEMORY_ENTRIES]
};

struct SenderTO
{
    uint8_t range;
    int maxTimesSent;
};

struct ReceiverTO
{
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
    LineageRestriction restrictToLineage;
};

union CommunicatorModeTO
{
    SenderTO sender;
    ReceiverTO receiver;
};

struct CommunicatorTO
{
    CommunicatorMode mode;
    CommunicatorModeTO modeData;
};

union CellTypeDataTO
{
    BaseTO base;
    DepotTO depot;
    SensorTO sensor;
    GeneratorTO generator;
    AttackerTO attacker;
    InjectorTO injector;
    MuscleTO muscle;
    DefenderTO defender;
    ReconnectorTO reconnector;
    DetonatorTO detonator;
    DigestorTO digestor;
    MemoryTO memory;
    CommunicatorTO communicator;
    VoidTO voidCell;
};

struct SignalTO
{
    float channels[NEURONS_PER_CELL];
    int numTimesSent;
};

struct SolidTO
{
    float energy;
};

struct FluidTO
{
    float energy;
    float glow;
};

struct FreeCellTO
{
    float energy;
    uint32_t age;
};

struct CellTO
{
    // General
    float usableEnergy;
    float rawEnergy;
    float frontAngle;  // May be invalid
    uint32_t age;
    CellState cellState;

    // Creature/Genome data
    uint64_t creatureIndex;
    uint16_t nodeIndex;
    uint16_t parentNodeIndex;
    uint16_t geneIndex;
    uint32_t concatenationIndex;
    uint8_t branchIndex;

    // Cell type data
    uint64_t neuralNetworkDataIndex;
    CellType cellType;
    CellTypeDataTO cellTypeData;
    bool constructorAvailable;  // If true, constructor holds valid data
    ConstructorTO constructor;  // Optional constructor data
    SignalTO signal;
    uint32_t activationTime;
    uint8_t lastUpdate;

    // Process data
    uint32_t headUpdateId;
    bool headCell;

    // Additional rendering data
    CellEvent event;
    uint8_t eventCounter;
    uint8_t signalChanges;
    float2 eventPos;
};

union ObjectTypeDataTO
{
    SolidTO solid;
    FluidTO fluid;
    FreeCellTO freeCell;
    CellTO cell;
};

struct ObjectTO
{
    // General
    uint64_t id;
    uint8_t numConnections;
    ConnectionTO connections[MAX_OBJECT_CONNECTIONS];
    float2 pos;
    float2 vel;
    float stiffness;
    uint8_t color;
    uint8_t flags;  // bit0 = isStatic, bit2 = sticky
    ObjectType type;
    ObjectTypeDataTO typeData;

    // Editing data
    uint8_t selected;

    __host__ __device__ __inline__ bool isStatic() const { return flags & 1; }
    __host__ __device__ __inline__ bool isSticky() const { return flags & 4; }

    __host__ __device__ __inline__ void setStatic(bool value) { flags = value ? (flags | 1) : (flags & ~1); }
    __host__ __device__ __inline__ void setSticky(bool value) { flags = value ? (flags | 4) : (flags & ~4); }
};

struct CreatureTO
{
    uint64_t id;
    uint64_t ancestorId;

    uint32_t generation;
    uint32_t numCells;

    uint64_t genomeArrayIndex;
    MutationState mutationState;

    uint32_t lineageId;
    float accumulatedMutations;

    // Process data
    uint32_t headUpdateId;

    // Temporary data
    uint64_t creatureIndexOnGpu;
};

struct TOs
{
    ArraySizesForTOs capacities;

    uint64_t* numObjects = nullptr;
    ObjectTO* objects = nullptr;
    uint64_t* numEnergyParticles = nullptr;
    EnergyTO* energyParticles = nullptr;
    uint64_t* numCreatures = nullptr;
    CreatureTO* creatures = nullptr;
    uint64_t* numGenomes = nullptr;
    GenomeTO* genomes = nullptr;
    uint64_t* numGenes = nullptr;
    GeneTO* genes = nullptr;
    uint64_t* numNodes = nullptr;
    NodeTO* nodes = nullptr;
    uint64_t* heapSize = nullptr;
    uint8_t* heap = nullptr;

    bool operator==(TOs const& other) const = default;
};
