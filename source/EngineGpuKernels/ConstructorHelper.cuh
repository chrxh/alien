#pragma once

#include "Object.cuh"

class ConstructorHelper
{
public:
    __inline__ __device__ static bool isStarting(Constructor const& constructor);
    __inline__ __device__ static bool isFinished(Constructor const& constructor, Genome const& genome);
    __inline__ __device__ static bool isFirstNode(Constructor const& constructor);
    __inline__ __device__ static bool isLastNode(Constructor const& constructor, Genome const& genome);
    __inline__ __device__ static bool isFirstConcatenation(Constructor const& constructor);
    __inline__ __device__ static bool isLastConcatenation(Constructor const& constructor, Genome const& genome);
    __inline__ __device__ static Gene* getCurrentGene(Constructor const& constructor, Genome const& genome);
    __inline__ __device__ static Node* getCurrentNode(Constructor const& constructor, Genome const& genome);
    __inline__ __device__ static bool isSeparation(Gene* gene);
    __inline__ __device__ static bool hasInfiniteConcatenations(Gene* gene);
    __inline__ __device__ static int getNumConstructedCellsOnBranch(Constructor const& constructor, Genome const& genome);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ bool ConstructorHelper::isStarting(Constructor const& constructor)
{
    return constructor.currentNodeIndex == 0 && constructor.currentBranch == 0 && constructor.currentConcatenation == 0;
}

__inline__ __device__ bool ConstructorHelper::isFinished(Constructor const& constructor, Genome const& genome)
{
    if (genome.numGenes == 0) {
        return true;
    }
    if (constructor.geneIndex >= genome.numGenes) {
        return true;
    }
    auto const& gene = getCurrentGene(constructor, genome);
    if (constructor.currentNodeIndex >= gene->numNodes) {
        return true;
    }
    if (constructor.currentConcatenation >= gene->numConcatenations) {
        return true;
    }
    if (isSeparation(gene)) {
        return false;
    }
    return constructor.currentBranch >= gene->numBranches;
}

__inline__ __device__ bool ConstructorHelper::isFirstNode(Constructor const& constructor)
{
    return constructor.currentNodeIndex == 0;
}

__inline__ __device__ bool ConstructorHelper::isLastNode(Constructor const& constructor, Genome const& genome)
{
    auto const& gene = getCurrentGene(constructor, genome);
    return constructor.currentNodeIndex == gene->numNodes - 1;
}

__inline__ __device__ bool ConstructorHelper::isFirstConcatenation(Constructor const& constructor)
{
    return constructor.currentConcatenation == 0;
}

__inline__ __device__ bool ConstructorHelper::isLastConcatenation(Constructor const& constructor, Genome const& genome)
{
    auto const& gene = getCurrentGene(constructor, genome);
    return constructor.currentConcatenation == gene->numConcatenations - 1;
}

__inline__ __device__ Gene* ConstructorHelper::getCurrentGene(Constructor const& constructor, Genome const& genome)
{
    CUDA_CHECK(constructor.geneIndex < genome.numGenes);
    return &genome.genes[constructor.geneIndex];
}

__inline__ __device__ Node* ConstructorHelper::getCurrentNode(Constructor const& constructor, Genome const& genome)
{
    auto gene = getCurrentGene(constructor, genome);
    return &gene->nodes[constructor.currentNodeIndex];
}

__inline__ __device__ bool ConstructorHelper::isSeparation(Gene* gene)
{
    return gene->separation;
}

__inline__ __device__ bool ConstructorHelper::hasInfiniteConcatenations(Gene* gene)
{
    return gene->numConcatenations == 0x7fffffff;
}

__inline__ __device__ int ConstructorHelper::getNumConstructedCellsOnBranch(Constructor const& constructor, Genome const& genome)
{
    auto gene = getCurrentGene(constructor, genome);
    return constructor.currentConcatenation * gene->numNodes + constructor.currentNodeIndex;
}
