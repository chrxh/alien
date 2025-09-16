#pragma once

#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/CellTypeConstants.h"

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

struct GenomeHeader
{
    ConstructorShape shape;
    int numBranches;
    bool separateConstruction;
    ConstructorAngleAlignment angleAlignment;
    float stiffness;
    float connectionDistance;
    int numRepetitions;
    float concatenationAngle1;
    float concatenationAngle2;
    float frontAngle;

    __inline__ __device__ bool hasInfiniteRepetitions() const { return numRepetitions == 0x7fffffff; }
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
    EnergyDistributionMode mode;
};

struct Constructor
{
    // Properties
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    uint16_t constructionActivationTime;
    float constructionAngle;

    // Genome data
    uint16_t geneIndex;

    // Process data
    uint64_t lastConstructedCellId;
    static auto constexpr LastConstructedCellId_NotSet = 0xffffffffffffffff;
    uint16_t currentNodeIndex;
    uint16_t currentConcatenation;
    uint8_t currentBranch;

     // Temp data
    bool isReady;
    Creature* offspring;    // Must be reset if separated construction is finished
};

struct Sensor
{
    uint32_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    float minDensity;
    int8_t minRange;          // < 0 = no restriction
    int8_t maxRange;          // < 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    SensorRestrictToCreatures restrictToCreatures;
};

struct Generator
{
    uint32_t autoTriggerInterval;
    GeneratorPulseType pulseType;
    uint32_t alternationInterval;  // Only for alternation type: 1 = alternate after each pulse, 2 = alternate after second pulse, etc.

    // Process data
    uint32_t numPulses;
};

struct Attacker
{
};

struct Injector
{
    InjectorMode mode;
    uint32_t counter;
};

struct AutoBending
{
    // Fixed data
    float maxAngleDeviation; // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1

    // Process data
    float initialAngle;
    float lastActualAngle;
    bool forward;  // Current direction
    float activation;
    uint8_t activationCountdown;
    bool impulseAlreadyApplied;
};

struct ManualBending
{
    // Fixed data
    float maxAngleDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1

    // Process data
    float initialAngle;
    float lastActualAngle;
    float lastAngleDelta;
    bool impulseAlreadyApplied;
};

struct AngleBending
{
    // Fixed data
    float maxAngleDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1

    // Process data
    float initialAngle;
};

struct AutoCrawling
{
    // Fixed data
    float maxDistanceDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1

    // Process data
    float initialDistance;
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
    float frontBackVelRatio;     // Between 0 and 1

    // Process data
    float initialDistance;
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
};

struct Defender
{
    DefenderMode mode;
};

struct Reconnector
{
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    ReconnectorRestrictToCreatures restrictToCreatures;
};

struct Detonator
{
    DetonatorState state;
    int32_t countdown;
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
};

struct SignalRestriction
{
    bool active;
    float baseAngle;
    float openingAngle;
};

struct Signal
{
    bool active;
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
    static auto constexpr AncestorId_NotSet = 0xffffffffffffffff;
    uint64_t ancestorId;

    uint32_t generation;
    uint32_t lineageId;
    uint32_t numCells;

    Genome genome;

    // Process data
    uint32_t frontAngleId;

    // Temporary data
    uint64_t creatureIndex;
    static auto constexpr CreatureIndex_NotSet = 0xffffffffffffffff;
};

struct Cell
{
    // General
    uint64_t id;
    uint8_t numConnections;
    CellConnection connections[MAX_CELL_BONDS];
    float2 pos;
    float2 vel;
    float energy;
    float stiffness;
    uint8_t color;
    float angleToFront;
    bool barrier;
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
    SignalRestriction signalRestriction;
    uint8_t signalRelaxationTime;
    Signal signal;
    uint32_t activationTime;
    CellTriggered cellTriggered;

    // Process data
    Signal futureSignal;
    uint16_t detectedByCreatureId;  // Only the first 16 bits from the creature id
    uint32_t frontAngleId;
    static auto constexpr FrontAngleId_NoUpdate = 10000.0f;
    bool isFrontAngleRefCell;

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

    __device__ __inline__ float getRefDistance(Cell* connectedCell)
    {
        for (int i = 0; i < numConnections; i++) {
            if (connections[i].cell == connectedCell) {
                return connections[i].distance;
            }
        }
        CUDA_CHECK(false);
        return 0;
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
        return Math::normalizedAngle(result, -180.0f);
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

    __device__ __inline__ void getLock()
    {
        while (1 == atomicExch(&locked, 1)) {}
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

template<>
struct HashFunctor<Cell*>
{
    __device__ __inline__ int operator()(Cell* const& cell)
    {
        return abs(static_cast<int>(cell->id));
    }
};
