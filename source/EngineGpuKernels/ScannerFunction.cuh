#pragma once
#include "cuda_runtime_api.h"

#include "EngineInterface/ElementaryTypes.h"

#include "SimulationData.cuh"
#include "Math.cuh"
#include "QuantityConverter.cuh"
#include "Token.cuh"
#include "Cell.cuh"

class ScannerFunction
{
public:
    __device__ static void processing(Token* token, SimulationData& data);

private:
    struct SpiralLookupResult
    {
        bool finish;
        Cell * cell;
        Cell * prevCell;
        Cell * prevPrevCell;
    };
    __device__ static SpiralLookupResult
    spiralLookupAlgorithm(int depth, Cell* cell, Cell* sourceCell, SimulationData& data);

    __device__ static int getConnectionIndex(Cell* cell, Cell* otherCell);
};

