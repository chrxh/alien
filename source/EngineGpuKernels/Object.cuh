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
    bool singleConstruction;
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

struct Activity
{
    float channels[MAX_CHANNELS];
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
    uint32_t offspringCreatureId;  //will be filled when self-replication starts
    uint32_t offspringMutationId;
    uint32_t stateFlags;  //bit 0: isConstructionBuilt

    //temp
    bool isComplete;

    __device__ __inline__ bool isConstructionBuilt() const { return (stateFlags & 0x1) != 0; }
    __device__ __inline__ void setConstructionBuilt(bool value) { stateFlags = (stateFlags & (~0x1)) | (value ? 0x1 : 0); }
};

struct SensorFunction
{
    SensorMode mode;
    float angle;
    float minDensity;
    uint8_t color;
    uint32_t targetedCreatureId;

    //process data
    float memoryChannel1;
    float memoryChannel2;
    float memoryChannel3;
};

struct NerveFunction
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
};

struct DefenderFunction
{
    DefenderMode mode;
};

struct ReconnectorFunction
{
    uint8_t color;
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
    NerveFunction nerve;
    AttackerFunction attacker;
    InjectorFunction injector;
    MuscleFunction muscle;
    DefenderFunction defender;
    ReconnectorFunction reconnector;
    DetonatorFunction detonator;
};

struct Cell
{
    uint64_t id;

    //general
    CellConnection connections[MAX_CELL_BONDS];
    float2 pos;
    float2 vel;
    uint8_t maxConnections;
    uint8_t numConnections;
    float energy;
    float stiffness;
    uint8_t color;
    bool barrier;
    uint32_t age;
    LivingState livingState;
    uint32_t creatureId;
    uint32_t mutationId;

    //cell function
    uint8_t executionOrderNumber;
    int8_t inputExecutionOrderNumber;
    bool outputBlocked;
    CellFunction cellFunction;
    CellFunctionData cellFunctionData;
    Activity activity;
    uint32_t activationTime;
    uint32_t attackProtection;

    //annotations
    CellMetadataDescription metadata;

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

    __device__ __inline__ bool isActive()
    {
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            if (abs(activity.channels[i]) > NEAR_ZERO) {
                return true;
            }
        }
        return false;
    }

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

