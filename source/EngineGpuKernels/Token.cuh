#pragma once
#include "Base.cuh"
#include "Definitions.cuh"
#include "ConstantMemory.cuh"

struct Token
{
    char memory[MAX_TOKEN_MEM_SIZE];
    Cell* sourceCell;
    Cell* cell;

    __inline__ __device__ int getTokenBranchNumber()
    {
        return static_cast<unsigned char>(memory[0]) % cudaSimulationParameters.cellMaxTokenBranchNumber;
    }

    __device__ __inline__ void setEnergy(float value)
    {
        _energy = value;
    }

    __device__ __inline__ void changeEnergy(float changeValue)
    {
        _energy += changeValue;
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

private:
    float _energy;
};
