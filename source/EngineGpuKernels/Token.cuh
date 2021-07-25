#pragma once
#include "Base.cuh"
#include "Definitions.cuh"
#include "ConstantMemory.cuh"

struct Token
{
    char memory[MAX_TOKEN_MEM_SIZE];
    Cell* sourceCell;
    Cell* cell;
    float energy;

    __inline__ __device__ int getTokenBranchNumber()
    {
        return static_cast<unsigned char>(memory[0]) % cudaSimulationParameters.cellMaxTokenBranchNumber;
    }
};
