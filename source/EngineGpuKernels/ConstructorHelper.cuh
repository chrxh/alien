#pragma once

#include "Object.cuh"

class ConstructorHelper
{
public:
    __inline__ __device__ static bool isSelfReplicator(Constructor const& constructor);
    __inline__ __device__ static bool isFinished(Cell* cell);   // cell must be of constructor type
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ bool ConstructorHelper::isSelfReplicator(Constructor const& constructor)
{
    return constructor.geneIndex == 0;
}

__inline__ __device__ bool ConstructorHelper::isFinished(Cell* cell)
{
    auto const& creature = cell->creature;
    if (creature == nullptr) {
        return false;
    }
    if (creature->numGenes == 0) {
        return true;
    }
    auto const& constructor = cell->cellTypeData.constructor;
    auto const& gene = creature->genes[constructor.geneIndex];
    if (gene.numBranches == 0) {
        return false;
    }
    return constructor.currentBranch >= gene.numBranches;
}
