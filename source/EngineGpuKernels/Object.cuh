#pragma once

#include <nppdefs.h>

#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/CellFunctionConstants.h"

#include "Base.cuh"

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

struct NeuronFunction
{
    struct NeuronState
    {
        float weights[MAX_CHANNELS * MAX_CHANNELS];
        float biases[MAX_CHANNELS];
    };

    NeuronState* neuronState;
    NeuronActivationFunction activationFunctions[MAX_CHANNELS];
};

struct TransmitterFunction
{
    EnergyDistributionMode mode;
};

struct ConstructorFunction
{
    //settings
    uint32_t activationMode;  //0 = manual, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    uint32_t constructionActivationTime;

    //genome
    uint16_t genomeSize;
    uint16_t numInheritedGenomeNodes;
    uint8_t* genome;
    uint32_t genomeGeneration;
    float constructionAngle1;
    float constructionAngle2;

    //process data
    uint64_t lastConstructedCellId;
    uint16_t genomeCurrentNodeIndex;
    uint16_t genomeCurrentRepetition;
    uint8_t currentBranch;
    uint32_t offspringCreatureId;  //will be filled when self-replication starts
    uint32_t offspringMutationId;

    //temp
    bool isReady;
};

struct SensorFunction
{
    float minDensity;
    int8_t minRange;          //< 0 = no restriction
    int8_t maxRange;          //< 0 = no restriction
    uint8_t restrictToColor;  //0 ... 6 = color restriction, 255 = no restriction
    SensorRestrictToMutants restrictToMutants;

    //process data
    float memoryChannel1;
    float memoryChannel2;
    float memoryChannel3;
    float memoryTargetX;
    float memoryTargetY;
};

struct OscillatorFunction
{
    uint8_t pulseMode;   //0 = none, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    uint8_t alternationMode;  //0 = none, 1 = alternate after each pulse, 2 = alternate after second pulse, 3 = alternate after third pulse, etc.
};

struct AttackerFunction
{
    EnergyDistributionMode mode;
};

struct InjectorFunction
{
    InjectorMode mode;
    uint32_t counter;
    uint16_t genomeSize;
    uint8_t* genome;
    uint32_t genomeGeneration;
};

struct MuscleFunction
{
    MuscleMode mode;
    MuscleBendingDirection lastBendingDirection;
    uint8_t lastBendingSourceIndex;
    float consecutiveBendingAngle;

    //additional rendering data
    float lastMovementX;
    float lastMovementY;
};

struct DefenderFunction
{
    DefenderMode mode;
};

struct ReconnectorFunction
{
    uint8_t restrictToColor;  //0 ... 6 = color restriction, 255 = no restriction
    ReconnectorRestrictToMutants restrictToMutants;
};

struct DetonatorFunction
{
    DetonatorState state;
    int32_t countdown;
};

union CellFunctionData
{
    NeuronFunction neuron;
    TransmitterFunction transmitter;
    ConstructorFunction constructor;
    SensorFunction sensor;
    OscillatorFunction oscillator;
    AttackerFunction attacker;
    InjectorFunction injector;
    MuscleFunction muscle;
    DefenderFunction defender;
    ReconnectorFunction reconnector;
    DetonatorFunction detonator;
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
    int numPrevCells;
    uint64_t prevCellIds[MAX_CELL_BONDS];
};

struct Cell
{
    uint64_t id;

    //general
    CellConnection connections[MAX_CELL_BONDS];
    float2 pos;
    float2 vel;
    uint8_t numConnections;
    float energy;
    float stiffness;
    uint8_t color;
    bool barrier;
    uint32_t age;
    LivingState livingState;
    uint32_t creatureId;
    uint32_t mutationId;
    uint8_t ancestorMutationId; //only the first 8 bits from ancestor mutation id
    float genomeComplexity;

    //cell function
    SignalRoutingRestriction signalRoutingRestriction;
    Signal signal;
    CellFunction cellFunction;
    CellFunctionData cellFunctionData;
    uint32_t activationTime;
    CellFunctionUsed cellFunctionUsed;

    //process data
    Signal futureSignal;
    uint16_t detectedByCreatureId;  //only the first 16 bits from the creature id

    //annotations
    CellMetadataDescription metadata;

    //additional rendering data
    CellEvent event;
    uint8_t eventCounter;
    float2 eventPos;

    //editing data
    uint8_t selected;  //0 = no, 1 = selected, 2 = cluster selected
    uint8_t detached;  //0 = no, 1 = yes

    //internal algorithm data
    int locked;  //0 = unlocked, 1 = locked
    int tag;
    float density;
    Cell* nextCell; //linked list for finding all overlapping cells
    int32_t scheduledOperationIndex;  // -1 = no operation scheduled
    float2 shared1; //variable with different meanings depending on context
    float2 shared2;

    //cluster data
    uint32_t clusterIndex;
    int32_t clusterBoundaries;  //1 = cluster occupies left boundary, 2 = cluster occupies upper boundary
    float2 clusterPos;
    float2 clusterVel;
    float clusterAngularMomentum;
    float clusterAngularMass;
    uint32_t numCellsInCluster;

    __device__ __inline__ uint8_t* getGenome()
    {
        if (cellFunction == CellFunction_Constructor) {
            return cellFunctionData.constructor.genome;
        }
        if (cellFunction == CellFunction_Injector) {
            return cellFunctionData.injector.genome;
        }
        CUDA_THROW_NOT_IMPLEMENTED();
        return nullptr;
    }

    __device__ __inline__ int getGenomeSize()
    {
        if (cellFunction == CellFunction_Constructor) {
            return cellFunctionData.constructor.genomeSize;
        }
        if (cellFunction == CellFunction_Injector) {
            return cellFunctionData.injector.genomeSize;
        }
        CUDA_THROW_NOT_IMPLEMENTED();
        return 0;
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

