#pragma once

#include "EngineInterface/ElementaryTypes.h"

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

struct CellConnection
{
    Cell* cell;
    float distance;
};

struct Cell
{
    uint64_t id;
    Cluster* cluster;
    float2 absPos;
    float2 vel;
    float2 force;
    float2 tempForce;
    float2 tempPos;
    float2 tempVel;

    int branchNumber;
    bool tokenBlocked;
    int maxConnections;
    int numConnections;
    CellConnection connections[MAX_CELL_BONDS];
    unsigned char numStaticBytes;
    char staticData[MAX_CELL_STATIC_BYTES];
    unsigned char numMutableBytes;
    char mutableData[MAX_CELL_MUTABLE_BYTES];
    int tokenUsages;
    CellMetadata metadata;

    //auxiliary data
    int locked;	//0 = unlocked, 1 = locked
    int alive;  //0 = dead, 1 == alive
    int tag;

    __inline__ __device__ void initProtectionCounter()
    {
        _protectionCounter = 0;
    }
                     
    __inline__ __device__ void activateProtectionCounter_safe()
    {
        atomicExch(&_protectionCounter, 14);
    }

    __inline__ __device__ void decrementProtectionCounter()
    {
        if (_protectionCounter > 0) {
            --_protectionCounter;
        }
    }

    __inline__ __device__ int getProtectionCounter()
    {
        return _protectionCounter;
    }

    __inline__ __device__ int getProtectionCounter_safe()
    {
        return atomicAdd(&_protectionCounter, 0);
    }

    __inline__ __device__ Enums::CellFunction::Type getCellFunctionType() const
    {
        return static_cast<Enums::CellFunction::Type>(static_cast<unsigned int>(_cellFunctionType)
            % Enums::CellFunction::_COUNTER);
    }

    __inline__ __device__ void setCellFunctionType(int value)
    {
        _cellFunctionType = value;
    }

    __device__ __inline__ void setEnergy_safe(float value)
    {
        atomicExch(&_energy, value);
    }

    __device__ __inline__ void changeEnergy_safe(float changeValue)
    {
        atomicAdd(&_energy, changeValue);
    }

    __device__ __inline__ float getEnergy_safe()
    {
        return atomicAdd(&_energy, 0);
    }

    __device__ __inline__ float getEnergy() const
    {
        return _energy;
    }

    __device__ __inline__ void repair()
    {
        if (isnan(_energy) || _energy < 0.0) {
            _energy = 0.01;
        }
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

    __device__ __inline__ bool isFused()
    {
        return _fused;
    }

    __device__ __inline__ void setFused(bool value)
    {
        _fused = value;
    }

private:
    int _cellFunctionType;
    float _energy;
    int _protectionCounter;

    //auxiliary data
    int _fused;
};

template<>
struct HashFunctor<Cell*>
{
    __device__ __inline__ int operator()(Cell* const& cell)
    {
        return abs(static_cast<int>(cell->id));
    }
};

