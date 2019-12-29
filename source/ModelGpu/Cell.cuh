#pragma once

#include "ModelBasic/ElementaryTypes.h"

#include "Base.cuh"
#include "Definitions.cuh"

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

    __inline__ __device__ Enums::CellFunction::Type getCellFunctionType() const
    {
        return static_cast<Enums::CellFunction::Type>(static_cast<unsigned int>(_cellFunctionType)
            % Enums::CellFunction::_COUNTER);
    }

    __inline__ __device__ void setCellFunctionType(int value)
    {
        _cellFunctionType = value;
    }

    __device__ __inline__ void setEnergy(float value)
    {
        atomicExch(&_energy, value);
    }

    __device__ __inline__ void changeEnergy(float changeValue)
    {
        atomicAdd(&_energy, changeValue);
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
    int _cellFunctionType;
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

