#pragma once

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/NeuralNetWeight.h>
#include <EngineInterface/SimulationParameters.h>

#include "CellProcessor.cuh"
#include "ConstantMemory.cuh"
#include "Genome.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class MutationProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);
    __inline__ __device__ static void applyMutations(SimulationData& data, Genome* genome);

private:
    __inline__ __device__ static void applyMutations_neurons(SimulationData& data, Genome* genome);
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
            applyMutations_neurons(data, mutatedGenome);

            // Update genome
            creature->genome = mutatedGenome;
        }
    }
}

__inline__ __device__ void MutationProcessor::applyMutations(SimulationData& data, Genome* genome)
{
    applyMutations_neurons(data, genome);
}

__inline__ __device__ void MutationProcessor::applyMutations_neurons(SimulationData& data, Genome* genome)
{
    NeuronMutationRate rates[2] = {genome->neuronMutationRate1, genome->neuronMutationRate2};

    for (int rateIndex = 0; rateIndex < 2; ++rateIndex) {
        auto const& rate = rates[rateIndex];
        if (rate.probability <= 0 || rate.weightSigma <= 0) {
            continue;
        }
        for (int geneIndex = 0; geneIndex < genome->numGenes; ++geneIndex) {
            auto& gene = genome->genes[geneIndex];
            for (int nodeIndex = 0; nodeIndex < gene.numNodes; ++nodeIndex) {
                auto& node = gene.nodes[nodeIndex];
                for (int neuronIndex = 0; neuronIndex < NEURONS_PER_CELL; ++neuronIndex) {
                    if (isRandomEvent(data, rate.probability)) {
                        for (int weightIndex = 0; weightIndex < NEURONS_PER_CELL; ++weightIndex) {
                            auto& weight = node.neuralNetwork.weights[neuronIndex * NEURONS_PER_CELL + weightIndex];

                            // Box-Muller transform for Gaussian random
                            float u1 = max(1.0e-7f, data.primaryNumberGen.random());
                            float u2 = data.primaryNumberGen.random();
                            float gaussian = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * Const::PI * u2);

                            float newValue = weight.getValue() + gaussian * rate.weightSigma;
                            newValue = max(-2.0f, min(2.0f, newValue));
                            weight = NeuralNetWeight(newValue);
                        }
                    }
                }
            }
        }
    }
}

__inline__ __device__ bool MutationProcessor::isRandomEvent(SimulationData& data, float probability)
{
    if (probability > 0.001f) {
        return data.primaryNumberGen.random() < probability;
    } else {
        return data.primaryNumberGen.random() < probability * 1000 && data.secondaryNumberGen.random() < 0.001f;
    }
}
