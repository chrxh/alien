#pragma once

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/SimulationParameters.h>

#include "ConstantMemory.cuh"
#include "CellProcessor.cuh"
#include "Genome.cuh"
#include "SimulationStatistics.cuh"
#include "SimulationData.cuh"

class MutationProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

    __inline__ __device__ static void applyMutations_neuralNetwork(SimulationData& data, Genome* genome);

private:
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
            applyMutations_neuralNetwork(data, mutatedGenome);
        }
    }
}

__inline__ __device__ void MutationProcessor::applyMutations_neuralNetwork(SimulationData& data, Genome* genome)
{
    auto mutationNeuralNetwork = cudaSimulationParameters.mutationNeuralNetwork.value;

    auto neuralNetworkMutationWeight =
        cudaSimulationParameters.customizeNeuralNetworkMutationsToggle ? cudaSimulationParameters.neuralNetworkMutationWeight : 0.2f;
    auto neuralNetworkMutationBias =
        cudaSimulationParameters.customizeNeuralNetworkMutationsToggle ? cudaSimulationParameters.neuralNetworkMutationBias : 0.2f;
    auto neuralNetworkMutationActivationFunction =
        cudaSimulationParameters.customizeNeuralNetworkMutationsToggle ? cudaSimulationParameters.neuralNetworkMutationActivationFunction : 0.05f;
    auto neuralNetworkMutationReinforcement =
        cudaSimulationParameters.customizeNeuralNetworkMutationsToggle ? cudaSimulationParameters.neuralNetworkMutationReinforcement : 1.05f;
    auto neuralNetworkMutationDamping =
        cudaSimulationParameters.customizeNeuralNetworkMutationsToggle ? cudaSimulationParameters.neuralNetworkMutationDamping : 1.05f;
    auto neuralNetworkMutationOffset =
        cudaSimulationParameters.customizeNeuralNetworkMutationsToggle ? cudaSimulationParameters.neuralNetworkMutationOffset : 0.05f;

    for (int i = 0, numGenes = genome->numGenes; i < numGenes; ++i) {
        auto gene = &genome->genes[i];
        for (int j = 0, numNodes = gene->numNodes; j < numNodes; ++j) {
            if (!isRandomEvent(data, mutationNeuralNetwork)) {
                continue;
            }

            auto node = &gene->nodes[j];
            auto neuronMutationType = data.primaryNumberGen.random(3);

            // Mutate weights
            for (int k = 0; k < MAX_CHANNELS * MAX_CHANNELS; ++k) {
                if (data.primaryNumberGen.random() < neuralNetworkMutationWeight) {
                    auto& property = node->neuralNetwork.weights[k];
                    if (neuronMutationType == 0) {
                        property *= neuralNetworkMutationReinforcement;
                    } else if (neuronMutationType == 1) {
                        property /= neuralNetworkMutationDamping;
                    } else if (neuronMutationType == 2) {
                        property += neuralNetworkMutationOffset;
                    } else if (neuronMutationType == 3) {
                        property -= neuralNetworkMutationOffset;
                    }
                }
            }

            // Mutate biases
            for (int k = 0; k < MAX_CHANNELS; ++k) {
                if (data.primaryNumberGen.random() < neuralNetworkMutationBias) {
                    auto& property = node->neuralNetwork.biases[k];
                    if (neuronMutationType == 0) {
                        property *= neuralNetworkMutationReinforcement;
                    } else if (neuronMutationType == 1) {
                        property /= neuralNetworkMutationDamping;
                    } else if (neuronMutationType == 2) {
                        property += neuralNetworkMutationOffset;
                    } else if (neuronMutationType == 3) {
                        property -= neuralNetworkMutationOffset;
                    }
                }
            }

            // Mutate activation functions
            for (int k = 0; k < MAX_CHANNELS; ++k) {
                if (data.primaryNumberGen.random() < neuralNetworkMutationActivationFunction) {
                    node->neuralNetwork.activationFunctions[k] = data.primaryNumberGen.randomByte();
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
