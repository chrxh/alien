#pragma once
#include "Base.cuh"
#include "Definitions.cuh"

struct Token
{
    float energy;
    char memory[MAX_TOKEN_MEM_SIZE];
    Cell* sourceCell;
    Cell* cell;
};
