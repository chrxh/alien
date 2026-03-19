#pragma once

#include <cuda_fp16.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/CudaSettings.h>
#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/NeuralNetWeight.h>

#include "Array.cuh"
#include "Base.cuh"
#include "Definitions.cuh"
#include "Genome.cuh"
#include "Math.cuh"

struct Energy
{
    uint64_t id;
    float2 pos;
    float2 vel;
    uint8_t color;
    float energy;
    Object* lastAbsorbedObject;  //could be invalid

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

struct ObjectConnection
{
    Object* object;
    float distance;
    float angleFromPrevious;
};

struct __align__(16) NeuralNet
{
    NeuralNetWeight weights[NEURONS_PER_CELL * NEURONS_PER_CELL];
    float biases[NEURONS_PER_CELL];
    ActivationFunction activationFunctions[NEURONS_PER_CELL];
    float connectionWeights[MAX_OBJECT_CONNECTIONS];
};

struct Base
{};

struct VoidCell
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
    uint16_t currentOffspring;
    uint8_t currentBranch;

    // Temp data
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
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
};

struct DetectCreature
{
    uint32_t minNumCells;      // 0 = no restriction
    uint32_t maxNumCells;      // 0 = no restriction
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
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
    uint16_t creatureIdPart;
    float2 pos;
};

struct Sensor
{
    bool autoTrigger;
    bool tagForAttackers;
    SensorMode mode;
    SensorModeData modeData;
    uint16_t minRange;
    uint16_t maxRange;

    // Process data
    bool lastMatchAvailable;
    SensorLastMatch lastMatch;
};

struct SquareSignal
{
    float amplitude;
    int period;
};

struct SawtoothSignal
{
    float amplitude;
    int period;
};

union GeneratorModeData
{
    SquareSignal squareSignal;
    SawtoothSignal sawtoothSignal;
};

struct Generator
{
    bool additive;
    float valueOffset;
    int timeOffset;
    GeneratorMode mode;
    GeneratorModeData modeData;

    // Process data
    uint32_t numPulses;
};

struct AttackFreeCell
{
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
};

struct AttackCreature
{};

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
};

struct ManualBending
{
    // Fixed data
    float maxAngleDeviation;     // Between 0 and 1
    float forwardBackwardRatio;  // Between 0 and 1

    // Process data
    float initialAngle;  // May be invalid
    float lastAngleDelta;
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
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
};

struct ReconnectCreature
{
    uint32_t minNumCells;     // 0 = no restriction
    uint32_t maxNumCells;     // 0 = no restriction
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
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

struct __align__(16) SignalEntry
{
    float channels[NEURONS_PER_CELL];
};

struct Memory
{
    MemoryMode mode;
    MemoryModeData modeData;

    uint8_t numSignalEntries;
    uint16_t channelBitMask;
    SignalEntry* signalEntries;  // Pointer to SignalEntry[MAX_CELL_MEMORY_ENTRIES] in heap
};

struct Sender
{
    uint8_t range;
    int maxTimesSent;
};

struct Receiver
{
    uint16_t restrictToColors;  // bitset: bit i set = color i allowed, 0x3ff = all colors
    LineageRestriction restrictToLineage;
};

union CommunicatorModeData
{
    Sender sender;
    Receiver receiver;
};

struct Communicator
{
    CommunicatorMode mode;
    CommunicatorModeData modeData;
};

union CellTypeData
{
    Base base;
    Depot depot;
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
    Communicator communicator;
    VoidCell voidCell;
};

struct __align__(16) Signal
{
    float channels[NEURONS_PER_CELL];
    int numTimesSent;
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
    float2 as_float2;
};

struct Creature
{
    uint64_t id;
    uint64_t ancestorId;  // May be invalid

    uint32_t generation;
    uint32_t numObjects;

    Genome* genome;
    MutationState mutationState;

    // Process data
    uint32_t frontAngleId;

    // Temporary data
    uint64_t creatureIndex;  // May be invalid
};

struct Structure
{
    float energy;

    // Cluster data
    uint32_t clusterIndex;
    int32_t clusterBoundaries;  // 1 = cluster occupies left boundary, 2 = cluster occupies upper boundary
    float2 clusterPos;
    float2 clusterVel;
    float clusterAngularMomentum;
    float clusterAngularMass;
    uint32_t numCellsInCluster;
};

struct Fluid
{
    float energy;
    float glow;
};

struct FreeCell
{
    float energy;
    uint32_t age;

    // Additional rendering data
    CellEvent event;
    uint8_t eventCounter;
    float2 eventPos;
};

struct Cell
{
    // General
    float usableEnergy;
    float rawEnergy;
    float reservedEnergy;
    float frontAngle;  // May be invalid
    uint32_t age;
    CellState cellState;

    // Creature/genome data
    Creature* creature;
    uint16_t nodeIndex;
    uint16_t parentNodeIndex;
    uint16_t geneIndex;

    // Cell type data
    NeuralNet* neuralNetwork;  // Not used for structure and base cells
    CellType cellType;
    CellTypeData cellTypeData;
    bool constructorAvailable;  // If true, constructor holds valid data
    Constructor constructor;    // Optional constructor data
    Signal signal;
    uint32_t activationTime;
    uint8_t lastUpdate;

    // Process data
    Signal futureSignal;
    uint32_t frontAngleId;
    bool headCell;

    // Additional rendering data
    CellEvent event;
    uint8_t eventCounter;
    uint8_t signalChanges;
    float2 eventPos;

    __device__ __inline__ bool isSameCreature(Cell* otherCell) { return otherCell->creature->id == this->creature->id; }

    __device__ __inline__ float getEnergy() const
    {
        auto result = usableEnergy + rawEnergy + reservedEnergy;
        if (cellType == CellType_Depot) {
            result += cellTypeData.depot.storedUsableEnergy;
        }
        return result;
    }
};

union ObjectTypeData
{
    Structure structure;
    Fluid fluid;
    FreeCell freeCell;
    Cell cell;
};

struct Object
{
    // General
    uint64_t id;
    uint8_t numConnections;
    ObjectConnection connections[MAX_OBJECT_CONNECTIONS];
    float2 pos;
    float2 vel;
    float stiffness;
    uint8_t color;
    bool fixed;
    bool sticky;
    ObjectType type;
    ObjectTypeData typeData;

    // Editing data
    uint8_t selected;  // 0 = no, 1 = selected, 2 = cluster selected
    uint8_t detached;  // 0 = no, 1 = yes

    // Internal algorithm data
    int locked;  // 0 = unlocked, 1 = locked
    TempValue tempValue;
    TempValue tempValue2;

    float density;
    Object* nextObject;               // Linked list for finding all overlapping cells
    int32_t scheduledOperationIndex;  // -1 = no operation scheduled
    float2 shared1;                   // Variable with different meanings depending on context
    float2 shared2;

    __device__ __inline__ float& getRefDistance(Object* connectedObject)
    {
        auto index = getConnectionIndex(connectedObject);
        return connections[index].distance;

        CUDA_CHECK(false);
        return tempValue.as_uint32_float.floatPart;  // Return some dummy in order to prevent compile error
    }

    __device__ __inline__ int getConnectionIndex(Object* connectedObject)
    {
        for (int i = 0; i < numConnections; i++) {
            if (connections[i].object == connectedObject) {
                return i;
            }
        }
        //CUDA_CHECK(false);
        return 0;
    }

    __device__ __inline__ ObjectConnection& getConnection(int index) { return connections[(index + numConnections) % numConnections]; }
    __device__ __inline__ Object* getConnectedObject(int index) { return connections[(index + numConnections) % numConnections].object; }
    __device__ __inline__ void increaseAngle(int index, float increment)
    {
        auto& angle1 = getConnection(index).angleFromPrevious;
        auto& angle2 = getConnection(index + 1).angleFromPrevious;
        if (angle1 + increment < 0 || angle2 - increment < 0) {
            return;
        }
        angle1 += increment;
        angle2 -= increment;
    }

    __device__ __inline__ float getEnergy() const
    {
        if (type == ObjectType_Cell) {
            return typeData.cell.getEnergy();
        } else if (type == ObjectType_FreeCell) {
            return typeData.freeCell.energy;
        } else if (type == ObjectType_Structure) {
            return typeData.structure.energy;
        } else if (type == ObjectType_Fluid) {
            return typeData.fluid.energy;
        } else {
            return 0;
        }
    }

    __device__ __inline__ float getMassForSPH() const
    {
        if (type == ObjectType_Fluid) {
            return 0.1f;
        } else {
            return 1.0f;
        }
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

    __device__ __inline__ float getAngelSpan(Object* connectedObject1, Object* connectedObject2)
    {
        auto connectionIndex1 = -1;
        auto connectionIndex2 = -1;
        for (int i = 0; i < numConnections; i++) {
            if (connections[i].object == connectedObject1) {
                connectionIndex1 = i;
            }
            if (connections[i].object == connectedObject2) {
                connectionIndex2 = i;
            }
        }
        if (connectionIndex1 == -1 || connectionIndex2 == -1) {
            return 0;
        }
        return getAngelSpan(connectionIndex1, connectionIndex2);
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

struct Entities
{
    Array<Object*> objects;
    Array<Energy*> energies;

    Heap heap;

    void init();
    void free();

    __device__ void saveNumEntries();
};

template <>
struct HashFunctor<Object*>
{
    __device__ __inline__ int operator()(Object* const& object) { return abs(static_cast<int>(object->id)); }
};
