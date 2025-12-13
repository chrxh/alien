#pragma once

#include "Genome.cuh"

class GenomeProcessor
{
public:
    __inline__ __device__ static float getRequiredEnergyForNodes(Gene* gene);  // No recursion
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ float GenomeProcessor::getRequiredEnergyForNodes(Gene* gene)
{
    auto result = 0.0f;
    for (int i = 0, j = gene->numNodes; i < j; ++i) {
        auto const& node = &gene->nodes[i];
        result += cudaSimulationParameters.normalCellEnergy.value[node->color];
        if (node->cellType == CellTypeGenome_Depot) {
            result += node->cellTypeData.depot.initialStoredUsableEnergy;
        }
    }
    return result;
}
