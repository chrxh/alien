#pragma once

#include "EngineInterface/Enums.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "AccessTOs.cuh"
#include "ConstantMemory.cuh"

struct CellMetadata
{
    unsigned char color;

    int nameLen;
    char* name;

    int descriptionLen;
    char* description;

    int sourceCodeLen;
    char* sourceCode;
};

struct CellConnection
{
    Cell* cell;
    float distance;
    float angleFromPrevious;
};

struct Cell
{
    uint64_t id;
    float2 absPos;
    float2 vel;

    int branchNumber;
    bool tokenBlocked;
    int maxConnections;
    int numConnections;
    CellConnection connections[MAX_CELL_BONDS];
    unsigned char numStaticBytes;
    char staticData[MAX_CELL_STATIC_BYTES];
    unsigned char numMutableBytes;
    char mutableData[MAX_CELL_MUTABLE_BYTES];
    int cellFunctionInvocations;
    CellMetadata metadata;
    float energy;
    int cellFunctionType;
    bool barrier;
    int age;

    //editing data
    int selected;   //0 = no, 1 = selected, 2 = cluster selected

    //temporary data
    int locked;	//0 = unlocked, 1 = locked
    int tag;
    float2 temp1;
    float2 temp2;
    float2 temp3;

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

    __inline__ __device__ Enums::CellFunction getCellFunctionType() const
    {
        return calcMod(cellFunctionType, Enums::CellFunction_Count);
    }

    __inline__ __device__ void initMemorySizes()
    {
        switch (getCellFunctionType()) {
        case Enums::CellFunction_Computation: {
            numStaticBytes = cudaSimulationParameters.cellFunctionComputerMaxInstructions * 3;
            numMutableBytes = cudaSimulationParameters.cellFunctionComputerCellMemorySize;
        } break;
        case Enums::CellFunction_Sensor: {
            numStaticBytes = 0;
            numMutableBytes = 5;
        } break;
        default: {
            numStaticBytes = 0;
            numMutableBytes = 0;
        }
        }
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

