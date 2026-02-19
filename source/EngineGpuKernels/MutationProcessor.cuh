#pragma once

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/SimulationParameters.h>
#include <EngineInterface/NeuralNetWeight.h>

#include "ConstantMemory.cuh"
#include "CellProcessor.cuh"
#include "Genome.cuh"
#include "SimulationStatistics.cuh"
#include "SimulationData.cuh"

class MutationProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);
    __inline__ __device__ static void applyMutations(SimulationData& data, Genome* genome);

private:
    __inline__ __device__ static void applyMutations_neuronWeights(SimulationData& data, Genome* genome);
    __inline__ __device__ static bool isRandomEvent(SimulationData& data, float probability);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void MutationProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    EntityFactory factory;
    factory.init(&data);

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        auto& creature = object->typeData.cell.creature;
        int origMutationState = atomicCAS(&creature->mutationState, MutationState_MutationInProgress, MutationState_Mutated);
        if (origMutationState == MutationState_MutationInProgress) {

            // Clone genome
            auto mutatedGenome = factory.cloneGenome(creature->genome);

            // Apply mutations to cloned genome
            applyMutations_neuronWeights(data, mutatedGenome);

            // Update genome
            creature->genome = mutatedGenome;
        }
    }
}

__inline__ __device__ void MutationProcessor::applyMutations(SimulationData& data, Genome* genome)
{
    applyMutations_neuronWeights(data, genome);
}

__inline__ __device__ void MutationProcessor::applyMutations_neuronWeights(SimulationData& data, Genome* genome)
{
}

__inline__ __device__ bool MutationProcessor::isRandomEvent(SimulationData& data, float probability)
{
    if (probability > 0.001f) {
        return data.primaryNumberGen.random() < probability;
    } else {
        return data.primaryNumberGen.random() < probability * 1000 && data.secondaryNumberGen.random() < 0.001f;
    }
}
