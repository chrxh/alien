#pragma once

#include <EngineInterface/ArraySizesForTO.h>
#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>

#include "GenomeTO.cuh"
#include <cuda_runtime.h>
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

struct NeuralNetworkTO
{
    float weights[MAX_CHANNELS * MAX_CHANNELS];
    float biases[MAX_CHANNELS];
    ActivationFunction activationFunctions[MAX_CHANNELS];
};

struct BaseTO
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

    // Genome data
    uint16_t geneIndex;

    // Process data
    uint64_t lastConstructedCellId;  // May be invalid
    uint16_t currentNodeIndex;
    uint16_t currentConcatenation;
    uint8_t currentBranch;
};

struct TelemetryTO
{};

struct DetectEnergyTO
{
    float minDensity;
};

struct DetectStructureTO
{};

struct DetectFreeCellTO
{
    float minDensity;
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
};

struct DetectCreatureTO
{
    uint32_t minNumCells;  // 0 = no restriction
    uint32_t maxNumCells;  // 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    LineageRestriction restrictToLineage;
};

union SensorModeTO
{
    TelemetryTO telemetry;
    DetectEnergyTO detectEnergy;
    DetectStructureTO detectStructure;
    DetectFreeCellTO detectFreeCell;
    DetectCreatureTO detectCreature;
};

struct SensorLastMatchTO
{
    uint64_t creatureId;
    float2 pos;
};

struct SensorTO
{
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    SensorMode mode;
    SensorModeTO modeData;
    uint16_t minRange;
    uint16_t maxRange;

    // Process data
    bool lastMatchAvailable;
    SensorLastMatchTO lastMatch;
};

struct GeneratorTO
{
    uint32_t autoTriggerInterval;
    GeneratorPulseType pulseType;
    uint32_t alternationInterval;  // Only for alternation type: 1 = alternate after each pulse, 2 = alternate after second pulse, etc.

    // Process data
    uint32_t numPulses;
};

struct AttackFreeCellTO
{
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
};

struct AttackCreatureTO
{
    uint32_t minNumCells;  // 0 = no restriction
    uint32_t maxNumCells;  // 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    LineageRestriction restrictToLineage;
};

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
    float activation;
    uint8_t activationCountdown;
    bool impulseAlreadyApplied;
};

struct ManualBendingTO
{
    // Fixed data
    float maxAngleDeviation;     // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1

    // Process data
    float initialAngle;  // May be invalid
    float lastAngleDelta;
    bool impulseAlreadyApplied;
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
    float activation;
    uint8_t activationCountdown;
    bool impulseAlreadyApplied;
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
    bool impulseAlreadyApplied;
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

struct ReconnectStructureTO
{};

struct ReconnectFreeCellTO
{
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
};

struct ReconnectCreatureTO
{
    uint32_t minNumCells;  // 0 = no restriction
    uint32_t maxNumCells;  // 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    LineageRestriction restrictToLineage;
};

union ReconnectorModeTO
{
    ReconnectStructureTO reconnectStructure;
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
    float channels[MAX_CHANNELS];
};

struct MemoryTO
{
    MemoryMode mode;
    MemoryModeDataTO modeData;

    uint8_t numSignalEntries;
    uint8_t channelBitMask;
    uint64_t signalEntriesDataIndex;  // Heap index to SignalEntryTO[MAX_CELL_MEMORY_ENTRIES]
};

struct SenderTO
{
    float range;
    int maxTimesSent;
};

struct ReceiverTO
{
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
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
    ConstructorTO constructor;
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
};

struct SignalRestrictionTO
{
    SignalRestrictionMode mode;
    float baseAngle;
    float openingAngle;
};

struct SignalTO
{
    float channels[MAX_CHANNELS];
    int numTimesSent;
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
    bool belongToCreature;
    uint64_t creatureIndex;
    uint16_t nodeIndex;
    uint16_t parentNodeIndex;
    uint16_t geneIndex;

    // Cell type data
    uint64_t neuralNetworkDataIndex;  // May be invalid (not used for structure and base cells)
    CellType cellType;
    CellTypeDataTO cellTypeData;
    SignalState signalState;
    SignalTO signal;  // For signalState == SignalState_Active
    SignalRestrictionTO signalRestriction;
    uint32_t activationTime;
    CellTriggered cellTriggered;

    // Process data
    uint32_t frontAngleId;
    bool headCell;
};

union ObjectTypeDataTO
{
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
    bool fixed;
    bool sticky;
    ObjectType type;
    ObjectTypeDataTO typeData;

    // Editing data
    uint8_t selected;
};

struct CreatureTO
{
    uint64_t id;
    uint64_t ancestorId;

    uint32_t generation;
    uint32_t lineageId;
    uint32_t numObjects;

    uint64_t genomeArrayIndex;

    // Process data
    uint32_t frontAngleId;

    // Temporary data
    uint64_t creatureIndexOnGpu;
};

struct TO
{
    ArraySizesForTO capacities;

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

    bool operator==(TO const& other) const = default;
};
