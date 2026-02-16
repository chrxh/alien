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

            // Update genome
            creature->genome = mutatedGenome;
        }
    }
}

__inline__ __device__ void MutationProcessor::applyMutations_neuralNetwork(SimulationData& data, Genome* genome)
{
    auto mutationNeuralNet = cudaSimulationParameters.mutationNeuralNet.value;

    auto neuralNetworkMutationWeight =
        cudaSimulationParameters.customizeNeuralNetMutationsToggle.value ? cudaSimulationParameters.neuralNetworkMutationWeight.value : 0.2f;
    auto neuralNetworkMutationBias =
        cudaSimulationParameters.customizeNeuralNetMutationsToggle.value ? cudaSimulationParameters.neuralNetworkMutationBias.value : 0.2f;
    auto neuralNetworkMutationActivationFunction =
        cudaSimulationParameters.customizeNeuralNetMutationsToggle.value ? cudaSimulationParameters.neuralNetworkMutationActivationFunction.value : 0.05f;
    auto neuralNetworkMutationReinforcement =
        cudaSimulationParameters.customizeNeuralNetMutationsToggle.value ? cudaSimulationParameters.neuralNetworkMutationReinforcement.value : 1.05f;
    auto neuralNetworkMutationDamping =
        cudaSimulationParameters.customizeNeuralNetMutationsToggle.value ? cudaSimulationParameters.neuralNetworkMutationDamping.value : 1.05f;
    auto neuralNetworkMutationOffset =
        cudaSimulationParameters.customizeNeuralNetMutationsToggle.value ? cudaSimulationParameters.neuralNetworkMutationOffset.value : 0.05f;

    for (int i = 0, numGenes = genome->numGenes; i < numGenes; ++i) {
        auto gene = &genome->genes[i];
        for (int j = 0, numNodes = gene->numNodes; j < numNodes; ++j) {
            if (!isRandomEvent(data, mutationNeuralNet)) {
                continue;
            }
            auto node = &gene->nodes[j];
            auto neuronMutationType = data.primaryNumberGen.random(3);

            for (int k = 0; k < MAX_CHANNELS * MAX_CHANNELS; ++k) {
                if (data.primaryNumberGen.random() < neuralNetworkMutationWeight) {
                    float property = node->neuralNetwork.weights[k].getValue();
                    if (neuronMutationType == 0) {
                        property *= neuralNetworkMutationReinforcement;
                    } else if (neuronMutationType == 1) {
                        property /= neuralNetworkMutationDamping;
                    } else if (neuronMutationType == 2) {
                        property += neuralNetworkMutationOffset;
                    } else if (neuronMutationType == 3) {
                        property -= neuralNetworkMutationOffset;
                    }
                    node->neuralNetwork.weights[k] = property;
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
