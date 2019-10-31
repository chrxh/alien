#pragma once

#include "Definitions.cuh"
#include "HashMap.cuh"
#include "Cell.cuh"

class Tagger
{
public:
    struct DynamicMemory
    {
        Cell** cellsToEvaluate;             //size of numCells
        Cell** cellsToEvaluateNextRound;    //size of numCells
        HashMap<Cell*, int> cellsEvaluated;
    };
    __inline__ __device__ static void tagComponent_blockCall(
        Cluster* cluster,
        Cell* startCell,
        int newTag,
        int freeTag,
        DynamicMemory& dynamicMemory    //expect shared memory
    );
};

__inline__ __device__ void Tagger::tagComponent_blockCall(
    Cluster* cluster,
    Cell* startCell,
    int cellTag,
    int freeTag,
    DynamicMemory& dynamicMemory)
{
    __shared__ int numCellsToEvaluate;
    __shared__ int numCellsToEvaluateNextRound;

    dynamicMemory.cellsEvaluated.init_blockCall();
    __syncthreads();

    if (0 == threadIdx.x) {
        numCellsToEvaluate = 1;
        numCellsToEvaluateNextRound = 0;
        dynamicMemory.cellsToEvaluate[0] = startCell;
        dynamicMemory.cellsEvaluated.insertOrAssign(startCell, 0);
    }
    __syncthreads();

    while (numCellsToEvaluate > 0) {
        auto const partition = calcPartition(numCellsToEvaluate, threadIdx.x, blockDim.x);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {

            auto const& cellToEvaluate = dynamicMemory.cellsToEvaluate[index];
            auto const numConnections = cellToEvaluate->numConnections;
            for (int i = 0; i < numConnections; ++i) {
                auto const& candidate = cellToEvaluate->connections[i];

                if (candidate->tag != freeTag) {
                    continue;
                }

                if (!dynamicMemory.cellsEvaluated.insertOrAssign(candidate, 0)) {
                    int origEvaluateIndex = atomicAdd(&numCellsToEvaluateNextRound, 1);
                    dynamicMemory.cellsToEvaluateNextRound[origEvaluateIndex] = candidate;
                    candidate->tag = cellTag;
                    __threadfence_block();
                }

            }
        }
        __syncthreads();

        if (0 == threadIdx.x) {
            numCellsToEvaluate = numCellsToEvaluateNextRound;
            numCellsToEvaluateNextRound = 0;
            swap(dynamicMemory.cellsToEvaluate, dynamicMemory.cellsToEvaluateNextRound);
        }
        __syncthreads();
    }

}
