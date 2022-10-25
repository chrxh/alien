#pragma once

#include "EngineInterface/Constants.h"
#include "EngineInterface/Enums.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "TOs.cuh"
#include "ConstantMemory.cuh"

struct CellMetadataDescription
{
    uint64_t nameSize;
    uint8_t* name;

    uint64_t descriptionSize;
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
        float bias[MAX_CHANNELS];
    };

    NeuronState* neuronState;
};

struct TransmitterFunction
{
};

struct RibosomeFunction
{
    Enums::ConstructionMode mode;
    uint64_t genomeSize;
    uint8_t* genome;
};

struct SensorFunction
{
    Enums::SensorMode mode;
    int color;
};

struct NerveFunction
{
};

struct AttackerFunction
{
};

struct InjectorFunction
{
    uint64_t genomeSize;
    uint8_t* genome;
};

struct MuscleFunction
{
};

union CellFunctionData
{
    NeuronFunction neuron;
    TransmitterFunction transmitter;
    RibosomeFunction ribosome;
    SensorFunction sensor;
    NerveFunction nerve;
    AttackerFunction digestion;
    InjectorFunction injector;
    MuscleFunction muscle;
};

struct Cell
{
    uint64_t id;
    CellConnection connections[MAX_CELL_BONDS];

    float2 absPos;
    float2 vel;
    int executionOrderNumber;
    int maxConnections;
    int numConnections;
    float energy;
    int color;
    bool barrier;
    int age;

    bool underConstruction;
    bool inputBlocked;
    bool outputBlocked;
    Enums::CellFunction cellFunction;
    CellFunctionData cellFunctionData;
    Activity activity;      //not inside cellFunctionData for easier handling and performance

    CellMetadataDescription metadata;

    //editing data
    int selected;   //0 = no, 1 = selected, 2 = cluster selected

    //temporary data for algorithms
    int activityFetched; //0 = no, 1 = yes
    int locked;	//0 = unlocked, 1 = locked
    int tag;
    float2 temp1;
    float2 temp2;

    //cluster data
    int clusterIndex;
    int clusterBoundaries;    //1 = cluster occupies left boundary, 2 = cluster occupies upper boundary
    float2 clusterPos;
    float2 clusterVel;
    float clusterAngularMomentum;
    float clusterAngularMass;
    int numCellsInCluster;

    __device__ __inline__ bool isDeleted() const { return energy == 0; }

    __device__ __inline__ void setDeleted()
    {
        energy = 0;
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

