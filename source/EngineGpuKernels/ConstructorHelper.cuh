#pragma once

#include "Object.cuh"

class ConstructorHelper
{
public:
    // cell parameter must be of constructor type

    __inline__ __device__ static bool isSelfReplicator(Constructor const& constructor);
    __inline__ __device__ static bool isFinished(Cell* cell);   
    __inline__ __device__ static bool isFirstNode(Cell* cell);
    __inline__ __device__ static bool isLastNode(Cell* cell);
    __inline__ __device__ static bool isFirstConcatenation(Cell* cell);
    __inline__ __device__ static bool isLastConcatenation(Cell* cell);
    __inline__ __device__ static Gene* getCurrentGene(Cell* cell);
    __inline__ __device__ static Node* getCurrentNode(Cell* cell);
    __inline__ __device__ static bool isSeparating(Gene* gene);
    __inline__ __device__ static bool hasInfiniteConcatenations(Gene* gene);
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
    if (creature->genome.numGenes == 0) {
        return true;
    }
    auto const& constructor = cell->cellTypeData.constructor;

    if (constructor.geneIndex >= creature->genome.numGenes) {
        return true;
    }
    auto const& gene = getCurrentGene(cell);
    if (isSeparating(gene) == 0) {
        return false;
    }
    if (constructor.currentNodeIndex >= gene->numNodes) {
        return true;
    }
    return constructor.currentBranch >= gene->numBranches;
}

__inline__ __device__ bool ConstructorHelper::isFirstNode(Cell* cell)
{
    return cell->cellTypeData.constructor.currentNodeIndex == 0;
}

__inline__ __device__ bool ConstructorHelper::isLastNode(Cell* cell)
{
    auto const& gene = getCurrentGene(cell);
    auto const& constructor = cell->cellTypeData.constructor;
    return constructor.currentNodeIndex == gene->numNodes - 1;
}

__inline__ __device__ bool ConstructorHelper::isFirstConcatenation(Cell* cell)
{
    return cell->cellTypeData.constructor.currentConcatenation == 0;
}

__inline__ __device__ bool ConstructorHelper::isLastConcatenation(Cell* cell)
{
    auto const& gene = getCurrentGene(cell);
    auto const& constructor = cell->cellTypeData.constructor;
    return constructor.currentConcatenation == gene->numConcatenations - 1;
}

__inline__ __device__ Gene* ConstructorHelper::getCurrentGene(Cell* cell)
{
    auto const& constructor = cell->cellTypeData.constructor;
    auto const& creature = cell->creature;
    CUDA_CHECK(constructor.geneIndex < creature->genome.numGenes);
    return &creature->genome.genes[constructor.geneIndex];
}

__inline__ __device__ Node* ConstructorHelper::getCurrentNode(Cell* cell)
{
    auto const& constructor = cell->cellTypeData.constructor;
    auto gene = getCurrentGene(cell);
    return &gene->nodes[constructor.currentNodeIndex];
}

__inline__ __device__ bool ConstructorHelper::isSeparating(Gene* gene)
{
    return gene->numBranches == 0;
}

__inline__ __device__ bool ConstructorHelper::hasInfiniteConcatenations(Gene* gene)
{
    return gene->numConcatenations == 0x7fffffff;
}
