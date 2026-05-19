#pragma once

#include "Entities.cuh"

class ConstructorHelper
{
public:
    __inline__ __device__ static bool isFinished(Object* constructorCell, Genome const& genome);
    __inline__ __device__ static Gene* getCurrentGene(Constructor const& constructor, Genome const& genome);
    __inline__ __device__ static bool hasInfiniteConcatenations(Constructor const& constructor);
    __inline__ __device__ static void
    getConstructorIndices(uint16_t& currentNodeIndex, uint32_t& currentConcatenation, uint8_t& currentBranch, Object* constructorCell, Genome const& genome);
    __inline__ __device__ static Object* getLastConstructedCell(Object* constructorCell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ bool ConstructorHelper::isFinished(Object* constructorCell, Genome const& genome)
{
    if (genome.numGenes == 0) {
        return true;
    }

    auto& constructor = constructorCell->typeData.cell.constructor;
    if (constructor.geneIndex >= genome.numGenes) {
        return true;
    }

    uint16_t currentNodeIndex;
    uint32_t currentConcatenation;
    uint8_t currentBranch;
    ConstructorHelper::getConstructorIndices(currentNodeIndex, currentConcatenation, currentBranch, constructorCell, genome);

    auto const& gene = getCurrentGene(constructor, genome);
    if (currentNodeIndex >= gene->numNodes) {
        return true;
    }
    if (currentConcatenation >= constructor.numConcatenations) {
        return true;
    }
    if (constructor.separation) {
        return false;
    }
    return currentBranch >= constructor.numBranches;
}

__inline__ __device__ Gene* ConstructorHelper::getCurrentGene(Constructor const& constructor, Genome const& genome)
{
    DEVICE_CHECK(constructor.geneIndex < genome.numGenes);
    return &genome.genes[constructor.geneIndex];
}

__inline__ __device__ bool ConstructorHelper::hasInfiniteConcatenations(Constructor const& constructor)
{
    return constructor.numConcatenations == 0x7fffffff;
}

__inline__ __device__ void ConstructorHelper::getConstructorIndices(
    uint16_t& currentNodeIndex,
    uint32_t& currentConcatenation,
    uint8_t& currentBranch,
    Object* constructorCell,
    Genome const& genome)
{
    auto lastConstructedCell = ConstructorHelper::getLastConstructedCell(constructorCell);
    if (lastConstructedCell) {
        currentNodeIndex = lastConstructedCell->typeData.cell.nodeIndex;
        currentConcatenation = lastConstructedCell->typeData.cell.concatenationIndex;
        currentBranch = lastConstructedCell->typeData.cell.branchIndex;

        auto& constructor = constructorCell->typeData.cell.constructor;
        auto const& gene = getCurrentGene(constructor, genome);
        ++currentNodeIndex;
        if (currentNodeIndex >= gene->numNodes) {
            currentNodeIndex = 0;
            currentConcatenation++;
            if (currentConcatenation >= constructor.numConcatenations) {
                currentConcatenation = 0;
                currentBranch++;
            }
        }
    } else {
        currentNodeIndex = 0;
        currentConcatenation = 0;
        currentBranch = 0;
    }
}

__inline__ __device__ Object* ConstructorHelper::getLastConstructedCell(Object* constructorCell)
{
    auto const& constructor = constructorCell->typeData.cell.constructor;
    if (constructor.lastConstructedCellId != VALUE_NOT_SET_UINT64) {
        for (int i = 0; i < constructorCell->numConnections; ++i) {
            auto const& connectedObject = constructorCell->connections[i].object;
            if (connectedObject->id == constructor.lastConstructedCellId) {
                return connectedObject;
            }
        }
    }
    return nullptr;
}
