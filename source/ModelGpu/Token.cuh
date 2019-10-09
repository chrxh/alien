#pragma once
#include "Base.cuh"
#include "Definitions.cuh"
#include "ConstantMemory.cuh"

struct Token
{
    float energy;
    char memory[MAX_TOKEN_MEM_SIZE];
    Cell* sourceCell;
    Cell* cell;

    __inline__ __device__ int getCellMaxConnections()
    {
        return static_cast<unsigned char>(memory[Enums::Constr::IN_CELL_MAX_CONNECTIONS])
            % (cudaSimulationParameters.cellMaxBonds + 1);
    }
};
