#pragma once

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>

#include "Base.cuh"
#include "Genome.cuh"
#include "Math.cuh"

struct Particle
{
    uint64_t id;
    float2 pos;
    float2 vel;
    uint8_t color;
    float energy;
    Cell* lastAbsorbedCell;  //could be invalid

    // Editing data
    uint8_t selected;  //0 = no, 1 = selected

    // Auxiliary data
    int locked;  //0 = unlocked, 1 = locked

    __device__ __inline__ bool tryLock()
    {
        auto result = 0 == atomicExch(&locked, 1);
        if (result) {
            __threadfence();
        }
        return result;
    }

    __device__ __inline__ void releaseLock()
    {
        __threadfence();
        atomicExch(&locked, 0);
    }
};

struct CellConnection
{
    Cell* cell;
    float distance;
    float angleFromPrevious;
};

struct NeuralNetwork
{
    float weights[MAX_CHANNELS * MAX_CHANNELS];
    float biases[MAX_CHANNELS];
    ActivationFunction activationFunctions[MAX_CHANNELS];
};

struct Base
{};

struct Depot
{
    float storageLimit;
    float storedUsableEnergy;
};

struct Constructor
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

    // Temp data
    bool isReady;
    Creature* offspring;  // Must be reset if separated construction is finished
};

struct Telemetry
{};

struct DetectEnergy
{
    float minDensity;
};

struct DetectStructure
{};

struct DetectFreeCell
{
    float minDensity;
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
};

struct DetectCreature
{
    uint32_t minNumCells;  // 0 = no restriction
    uint32_t maxNumCells;  // 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    LineageRestriction restrictToLineage;
};

union SensorModeData
{
    Telemetry telemetry;
    DetectEnergy detectEnergy;
    DetectStructure detectStructure;
    DetectFreeCell detectFreeCell;
    DetectCreature detectCreature;
};

struct SensorLastMatch
{
    uint64_t creatureId;
    float2 pos;
};

struct Sensor
{
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    SensorMode mode;
    SensorModeData modeData;
    uint16_t minRange;
    uint16_t maxRange;

    // Process data
    bool lastMatchAvailable;
    SensorLastMatch lastMatch;
};

struct Generator
{
    uint32_t autoTriggerInterval;
    GeneratorPulseType pulseType;
    uint32_t alternationInterval;  // Only for alternation type: 1 = alternate after each pulse, 2 = alternate after second pulse, etc.

    // Process data
    uint32_t numPulses;
};

struct AttackFreeCell
{
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
};

struct AttackCreature
{
    uint32_t minNumCells;  // 0 = no restriction
    uint32_t maxNumCells;  // 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    LineageRestriction restrictToLineage;
};

union AttackerModeData
{
    AttackFreeCell attackFreeCell;
    AttackCreature attackCreature;
};

struct Attacker
{
    AttackerMode mode;
    AttackerModeData modeData;
};

struct Injector
{
    uint16_t geneIndex;
};

struct AutoBending
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

struct ManualBending
{
    // Fixed data
    float maxAngleDeviation;     // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1

    // Process data
    float initialAngle;  // May be invalid
    float lastAngleDelta;
    bool impulseAlreadyApplied;
};

struct AngleBending
{
    // Fixed data
    float maxAngleDeviation;         // Between 0 and 1
    float attractionRepulsionRatio;  // Between 0 and 1

    // Process data
    float initialAngle;  // May be invalid
};

struct AutoCrawling
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

struct ManualCrawling
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

struct DirectMovement
{};

union MuscleModeData
{
    AutoBending autoBending;
    ManualBending manualBending;
    AngleBending angleBending;
    AutoCrawling autoCrawling;
    ManualCrawling manualCrawling;
    DirectMovement directMovement;
};

struct Muscle
{
    // Fixed data
    MuscleMode mode;
    MuscleModeData modeData;

    // Additional rendering data
    float lastMovementX;
    float lastMovementY;

    __inline__ __device__ bool isBendingMuscle() const
    {
        return mode == MuscleMode_AutoBending || mode == MuscleMode_ManualBending || mode == MuscleMode_AngleBending;
    }
};

struct Defender
{
    DefenderMode mode;
};

struct ReconnectStructure
{};

struct ReconnectFreeCell
{
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
};

struct ReconnectCreature
{
    uint32_t minNumCells;  // 0 = no restriction
    uint32_t maxNumCells;  // 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    LineageRestriction restrictToLineage;
};

union ReconnectorModeData
{
    ReconnectStructure reconnectStructure;
    ReconnectFreeCell reconnectFreeCell;
    ReconnectCreature reconnectCreature;
};

struct Reconnector
{
    ReconnectorMode mode;
    ReconnectorModeData modeData;
};

struct Detonator
{
    DetonatorState state;
    int32_t countdown;
};

struct Digestor
{
    float rawEnergyConductivity;  // Between 0 and 1
};

struct SignalDelay
{
    uint8_t delay;
    uint8_t numSignalEntriesInitialized;
    uint8_t ringBufferIndex;
};

struct SignalRecorder
{
    bool readOnly;
    SignalRecorderState state;
    uint8_t numWrittenSignalEntries;
    uint8_t numReadSignalEntries;
};

struct SignalStorage
{
    bool readOnly;
};

struct SignalIntegrator
{
    float newSignalWeight;  // Between 0 and 1
};

union MemoryModeData
{
    SignalDelay signalDelay;
    SignalRecorder signalRecorder;
    SignalStorage signalStorage;
    SignalIntegrator signalIntegrator;
};

struct SignalEntry
{
    float channels[MAX_CHANNELS];
};

struct Memory
{
    MemoryMode mode;
    MemoryModeData modeData;

    uint8_t numSignalEntries;
    SignalEntry* signalEntries;  // Pointer to SignalEntry[MAX_CELL_MEMORY_ENTRIES] in heap
};

union CellTypeData
{
    Base base;
    Depot depot;
    Constructor constructor;
    Sensor sensor;
    Generator generator;
    Attacker attacker;
    Injector injector;
    Muscle muscle;
    Defender defender;
    Reconnector reconnector;
    Detonator detonator;
    Digestor digestor;
    Memory memory;
};

struct SignalRestriction
{
    bool active;
    float baseAngle;
    float openingAngle;
};

struct Signal
{
    float channels[MAX_CHANNELS];
};

struct uint32_float
{
    uint32_t uint32Part;
    float floatPart;
};
union TempValue
{
    uint64_t as_uint64;
    uint32_float as_uint32_float;
};

struct Creature
{
    uint64_t id;
    uint64_t ancestorId;  // May be invalid

    uint32_t generation;
    uint32_t lineageId;
    uint32_t numCells;

    Genome* genome;

    // Process data
    uint32_t frontAngleId;

    // Temporary data
    uint64_t creatureIndex;  // May be invalid
};

struct Cell
{
    // General
    uint64_t id;
    uint8_t numConnections;
    CellConnection connections[MAX_CELL_BONDS];
    float2 pos;
    float2 vel;
    float usableEnergy;
    float rawEnergy;
    float stiffness;
    uint8_t color;
    float frontAngle;  // May be invalid
    bool fixed;
    bool sticky;
    uint32_t age;
    CellState cellState;

    // Creature/genome data
    Creature* creature;
    uint16_t nodeIndex;
    uint16_t parentNodeIndex;
    uint16_t geneIndex;

    // Cell type data
    NeuralNetwork* neuralNetwork;  // Not used for structure and base cells
    CellType cellType;
    CellTypeData cellTypeData;
    SignalState signalState;  // For signalState == SignalState_Active
    Signal signal;
    SignalRestriction signalRestriction;
    uint32_t activationTime;
    CellTriggered cellTriggered;

    // Process data
    SignalState futureSignalState;
    Signal futureSignal;
    uint32_t frontAngleId;
    bool headCell;

    // Additional rendering data
    CellEvent event;
    uint8_t eventCounter;
    float2 eventPos;

    // Editing data
    uint8_t selected;  // 0 = no, 1 = selected, 2 = cluster selected
    uint8_t detached;  // 0 = no, 1 = yes

    // Internal algorithm data
    int locked;  // 0 = unlocked, 1 = locked
    TempValue tempValue;

    float density;
    Cell* nextCell;                   // Linked list for finding all overlapping cells
    int32_t scheduledOperationIndex;  // -1 = no operation scheduled
    float2 shared1;                   // Variable with different meanings depending on context
    float2 shared2;

    // Cluster data
    uint32_t clusterIndex;
    int32_t clusterBoundaries;  // 1 = cluster occupies left boundary, 2 = cluster occupies upper boundary
    float2 clusterPos;
    float2 clusterVel;
    float clusterAngularMomentum;
    float clusterAngularMass;
    uint32_t numCellsInCluster;

    __device__ __inline__ bool isSameCreature(Cell* otherCell)
    {
        return (otherCell->creature != nullptr && this->creature != nullptr && otherCell->creature->id == this->creature->id)
            || (otherCell->creature == nullptr && this->creature == nullptr);
    }

    __device__ __inline__ float& getRefDistance(Cell* connectedCell)
    {
        auto index = getConnectionIndex(connectedCell);
        return connections[index].distance;

        CUDA_CHECK(false);
        return tempValue.as_uint32_float.floatPart;  // Return some dummy in order to prevent compile error
    }

    __device__ __inline__ int getConnectionIndex(Cell* connectedCell)
    {
        for (int i = 0; i < numConnections; i++) {
            if (connections[i].cell == connectedCell) {
                return i;
            }
        }
        //CUDA_CHECK(false);
        return 0;
    }

    __device__ __inline__ CellConnection& getConnection(int index) { return connections[(index + numConnections) % numConnections]; }
    __device__ __inline__ Cell* getConnectedCell(int index) { return connections[(index + numConnections) % numConnections].cell; }
    __device__ __inline__ void increaseAngle(int index, float increment) {
        auto& angle1 = getConnection(index).angleFromPrevious;
        auto& angle2 = getConnection(index + 1).angleFromPrevious;
        if (angle1 + increment < 0 || angle2 - increment < 0) {
            return;
        }
        angle1 += increment;
        angle2 -= increment;
    }

    __device__ __inline__ float getAngelSpan(int connectionIndex1, int connectionIndex2)
    {
        if ((connectionIndex1 - connectionIndex2 + numConnections) % numConnections == 0) {
            return 0;
        }
        auto result = 0.0f;
        for (int i = connectionIndex1 + 1; i < connectionIndex1 + numConnections; i++) {
            auto index = i % numConnections;
            result += connections[index].angleFromPrevious;
            if (index == connectionIndex2) {
                break;
            }
        }
        return Math::getNormalizedAngle(result, 0.0f);
    }

    __device__ __inline__ float getAngelSpan(Cell* connectedCell1, Cell* connectedCell2)
    {
        auto connectionIndex1 = -1;
        auto connectionIndex2 = -1;
        for (int i = 0; i < numConnections; i++) {
            if (connections[i].cell == connectedCell1) {
                connectionIndex1 = i;
            }
            if (connections[i].cell == connectedCell2) {
                connectionIndex2 = i;
            }
        }
        if (connectionIndex1 == -1 || connectionIndex2 == -1) {
            return 0;
        }
        return getAngelSpan(connectionIndex1, connectionIndex2);
    }

    __device__ __inline__ float getEnergy() const
    {
        auto result = usableEnergy + rawEnergy;
        if (cellType == CellType_Depot) {
            result += cellTypeData.depot.storedUsableEnergy;
        }
        return result;
    }


    __device__ __inline__ void getLock()
    {
        while (1 == atomicExch(&locked, 1)) {
        }
    }

    __device__ __inline__ bool tryLock()
    {
        auto result = 0 == atomicExch(&locked, 1);
        if (result) {
            __threadfence();
        }
        return result;
    }

    __device__ __inline__ void releaseLock()
    {
        __threadfence();
        atomicExch(&locked, 0);
    }
};

template <>
struct HashFunctor<Cell*>
{
    __device__ __inline__ int operator()(Cell* const& cell) { return abs(static_cast<int>(cell->id)); }
};
