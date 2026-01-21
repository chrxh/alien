#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "CellProcessor.cuh"
#include "Genome.cuh"
#include "SimulationStatistics.cuh"

class MutationProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void MutationProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        auto& creature = object->typeData.cell.creature;
        int origMutationState = atomicCAS(&creature->mutationState, MutationState_MutationInProgress, MutationState_Mutated);
        if (origMutationState == MutationState_MutationInProgress) {
            
            // Apply mutations to genome
        }
    }
}
