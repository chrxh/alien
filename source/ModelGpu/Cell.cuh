#pragma once

#include "Base.cuh"
#include "Definitions.cuh"

struct CellMetadata
{
    unsigned char color;
};

struct Cell
{
    uint64_t id;
    Cluster* cluster;
    float2 relPos;
    float2 absPos;
    float2 vel;
    int branchNumber;
    bool tokenBlocked;
    int maxConnections;
    int numConnections;
    Cell* connections[MAX_CELL_BONDS];
    int cellFunctionType;
    unsigned char numStaticBytes;
    char staticData[MAX_CELL_STATIC_BYTES];
    unsigned char numMutableBytes;
    char mutableData[MAX_CELL_MUTABLE_BYTES];
    int age;
    CellMetadata metadata;

    //auxiliary data
    int locked;	//0 = unlocked, 1 = locked
    int protectionCounter;
    int alive;  //0 = dead, 1 == alive
    int tag;

    __device__ __inline__ void setEnergy(float value)
    {
        if (value < 0) {
            printf("Cell::setEnergy negative, value: %f\n", value);
            while (true) {}
        }
        if (isnan(value)) {
            printf("Cell::setEnergy nan, value: %f\n", value);
            while (true) {}
        }
        atomicExch(&_energy, value);
    }

    __device__ __inline__ void changeEnergy(float changeValue, int parameter = -1)
    {
        auto const origValue = atomicAdd(&_energy, changeValue);
        if (getEnergy() < 0) {
            printf("Cell::changeEnergy negative, before: %f, after: %f, change: %f, parameter: %d\n", origValue, getEnergy(), changeValue, parameter);
            while (true) {}
        }
        if (isnan(getEnergy())) {
            printf("Cell::changeEnergy nan, before: %f, after: %f, change: %f, parameter: %d\n", origValue, getEnergy(), changeValue, parameter);
            while (true) {}
        }
    }

    __device__ __inline__ float getEnergy()
    {
        return atomicAdd(&_energy, 0);
    }

    __device__ __inline__ void getLock()
    {
        while (1 == atomicExch(&locked, 1)) {}
    }

    __device__ __inline__ bool tryLock()
    {
        return 0 == atomicExch(&locked, 1);
    }

    __device__ __inline__ void releaseLock()
    {
        atomicExch(&locked, 0);
    }

private:
    float _energy;
};

template<>
struct HashFunctor<Cell*>
{
    __device__ __inline__ int operator()(Cell* const& cell)
    {
        return abs(static_cast<int>(cell->id));
    }
};

