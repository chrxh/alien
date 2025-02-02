#pragma once

#include <optional>

#include <nppdefs.h>

#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/CellTypeConstants.h"

#include "Base.cuh"
#include "Math.cuh"

struct Particle
{
    uint64_t id;
    float2 absPos;
    float2 vel;
    uint8_t color;
    float energy;
    Cell* lastAbsorbedCell;  //could be invalid

    //editing data
    uint8_t selected;  //0 = no, 1 = selected

    //auxiliary data
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
    ConstructionShape shape;
    int numBranches;
    bool separateConstruction;
    ConstructorAngleAlignment angleAlignment;
    float stiffness;
    float connectionDistance;
    int numRepetitions;
    float concatenationAngle1;
    float concatenationAngle2;
    float frontAngle;

    __inline__ __device__ bool hasInfiniteRepetitions() const { return numRepetitions == NPP_MAX_32S; }
};

struct CellMetadataDescription
{
    uint16_t nameSize;
    uint8_t* name;

    uint16_t descriptionSize;
    uint8_t* description;
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

struct BaseType
{
};

struct TransmitterType
{
    EnergyDistributionMode mode;
};

struct ConstructorType
{
    // Properties
    uint8_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    uint16_t constructionActivationTime;

    // Genome data
    uint16_t genomeSize;
    uint16_t numInheritedGenomeNodes;
    uint8_t* genome;
    uint32_t genomeGeneration;
    float constructionAngle1;
    float constructionAngle2;

    // Process data
    uint64_t lastConstructedCellId;
    uint16_t genomeCurrentNodeIndex;
    uint16_t genomeCurrentRepetition;
    uint8_t genomeCurrentBranch;
    uint32_t offspringCreatureId;  // Will be filled when self-replication starts
    uint32_t offspringMutationId;

    // Temp data
    bool isReady;
};

struct SensorType
{
    uint8_t autoTriggerInterval;  // 0 = manual (triggered by signal), > 0 = auto trigger
    float minDensity;
    int8_t minRange;          // < 0 = no restriction
    int8_t maxRange;          // < 0 = no restriction
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    SensorRestrictToMutants restrictToMutants;
};

struct OscillatorType
{
    uint8_t autoTriggerInterval;
    uint8_t alternationInterval;  // 0 = none, 1 = alternate after each pulse, 2 = alternate after second pulse, 3 = alternate after third pulse, etc.

    // Process data
    int numPulses;
};

struct AttackerType
{
    EnergyDistributionMode mode;
};

struct InjectorType
{
    InjectorMode mode;
    uint32_t counter;
    uint16_t genomeSize;
    uint8_t* genome;
    uint32_t genomeGeneration;
};

struct AutoBending
{
    // Fixed data
    float maxAngleDeviation; // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1

    // Process data
    float initialAngle;
    float lastAngle;
    bool forward;  // Current direction
    float activation;
    uint8_t activationCountdown;
    bool impulseAlreadyApplied;
};

struct ManualBending
{
    //// Fixed data
    //float maxAngleDeviation;  // Between 0 and 1
    //float frontBackVelRatio;  // Between 0 and 1

    //// Process data
    //float initialAngle;
    //float lastAngle;
    //bool forward;  // Current direction
    //float activation;
    //uint8_t activationCountdown;
    //bool impulseAlreadyApplied;
};

struct AngleBending
{
    //// Fixed data
    //float maxAngleDeviation;  // Between 0 and 1
    //float frontBackVelRatio;  // Between 0 and 1

    //// Process data
};

struct AutoCrawling
{
    // Fixed data
    float maxDistanceDeviation;  // Between 0 and 1
    float frontBackVelRatio;  // Between 0 and 1

    // Process data
    float initialAngle;
    float lastAngle;
    bool forward;  // Current direction
    float activation;
    uint8_t activationCountdown;
    bool impulseAlreadyApplied;
};

union MuscleModeData
{
    AutoBending autoBending;
    ManualBending manualBending;
    AngleBending angleBending;
    AutoCrawling crawling;
};

struct MuscleType
{
    // Fixed data
    MuscleMode mode;
    MuscleModeData modeData;

    // Temp
    float frontAngle;  // Can be removed in new genome model

    // Additional rendering data
    float lastMovementX;
    float lastMovementY;
};

struct DefenderType
{
    DefenderMode mode;
};

struct ReconnectorType
{
    uint8_t restrictToColor;  // 0 ... 6 = color restriction, 255 = no restriction
    ReconnectorRestrictToMutants restrictToMutants;
};

struct DetonatorType
{
    DetonatorState state;
    int32_t countdown;
};

union CellTypeData
{
    BaseType base;
    TransmitterType transmitter;
    ConstructorType constructor;
    SensorType sensor;
    OscillatorType oscillator;
    AttackerType attacker;
    InjectorType injector;
    MuscleType muscle;
    DefenderType defender;
    ReconnectorType reconnector;
    DetonatorType detonator;
};

struct SignalRoutingRestriction
{
    bool active;
    float baseAngle;
    float openingAngle;
};

struct Signal
{
    bool active;
    float channels[MAX_CHANNELS];
    SignalOrigin origin;
    float targetX;
    float targetY;
};

struct Cell
{
    // General
    uint64_t id;
    CellConnection connections[MAX_CELL_BONDS];
    float2 pos;
    float2 vel;
    uint8_t numConnections;
    float energy;
    float stiffness;
    uint8_t color;
    float absAngleToConnection0;
    bool barrier;
    uint32_t age;
    LivingState livingState;
    uint32_t creatureId;
    uint32_t mutationId;
    uint8_t ancestorMutationId; // Only the first 8 bits from ancestor mutation id
    float genomeComplexity;
    uint16_t genomeNodeIndex;

    // Cell type data
    NeuralNetwork* neuralNetwork;
    CellType cellType;
    CellTypeData cellTypeData;
    SignalRoutingRestriction signalRoutingRestriction;
    uint8_t signalRelaxationTime;
    Signal signal;
    uint32_t activationTime;
    CellTriggered cellTypeUsed;

    // Process data
    Signal futureSignal;
    uint16_t detectedByCreatureId;  // Only the first 16 bits from the creature id

    // Annotations
    CellMetadataDescription metadata;

    // Additional rendering data
    CellEvent event;
    uint8_t eventCounter;
    float2 eventPos;

    // Editing data
    uint8_t selected;  // 0 = no, 1 = selected, 2 = cluster selected
    uint8_t detached;  // 0 = no, 1 = yes

    // Internal algorithm data
    int locked;  // 0 = unlocked, 1 = locked
    int tag;
    float density;
    Cell* nextCell; // Linked list for finding all overlapping cells
    int32_t scheduledOperationIndex;  // -1 = no operation scheduled
    float2 shared1; // Variable with different meanings depending on context
    float2 shared2;

    // Cluster data
    uint32_t clusterIndex;
    int32_t clusterBoundaries;  // 1 = cluster occupies left boundary, 2 = cluster occupies upper boundary
    float2 clusterPos;
    float2 clusterVel;
    float clusterAngularMomentum;
    float clusterAngularMass;
    uint32_t numCellsInCluster;

    __device__ __inline__ uint8_t* getGenome()
    {
        if (cellType == CellType_Constructor) {
            return cellTypeData.constructor.genome;
        }
        if (cellType == CellType_Injector) {
            return cellTypeData.injector.genome;
        }
        CUDA_CHECK(false);
        return nullptr;
    }

    __device__ __inline__ int getGenomeSize()
    {
        if (cellType == CellType_Constructor) {
            return cellTypeData.constructor.genomeSize;
        }
        if (cellType == CellType_Injector) {
            return cellTypeData.injector.genomeSize;
        }
        CUDA_CHECK(false);
        return 0;
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

