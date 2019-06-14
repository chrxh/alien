#pragma once

#include "Base.cuh"
#include "Cell.cuh"

struct Cluster
{
    uint64_t id;
    float2 pos;
    float2 vel;
    float angle;
    float angularVel;
    float angularMass;
    int numCellPointers;
    Cell** cellPointers;
    int numTokenPointers;
    Token** tokenPointers;

    //auxiliary data
    bool decompositionRequired;
    int locked;	//0 = unlocked, 1 = locked
    Cluster* clusterToFuse;

    __device__ __inline__ void tagCellByIndex_blockCall(BlockData const& blockData)
    {
        for (auto cellIndex = blockData.startIndex; cellIndex <= blockData.endIndex; ++cellIndex) {
            Cell& cell = *cellPointers[cellIndex];
            cell.tag = cellIndex;
        }
        __syncthreads();
    }
};
