#pragma once

#include <cooperative_groups.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/NeuralNetWeight.h>
#include <EngineInterface/SimulationParameters.h>

#include "CellProcessor.cuh"
#include "ConstantMemory.cuh"
#include "Genome.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

namespace cg_mutation = cooperative_groups;

class MutationProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);
    __inline__ __device__ static void applyMutations(SimulationData& data, Genome* genome);

private:
    __inline__ __device__ static void applyMutations_neurons(SimulationData& data, Genome* genome);
    __inline__ __device__ static void applyMutations_connections(SimulationData& data, Genome* genome);
    __inline__ __device__ static void applyMutations_lineages(SimulationData& data, Genome* genome);
    __inline__ __device__ static void applyMutations_meta(SimulationData& data, Genome* genome);
    __inline__ __device__ static float generateGaussian(SimulationData& data);
    __inline__ __device__ static bool isRandomEvent(SimulationData& data, float probability);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void MutationProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    CUDA_CHECK(blockDim.x == NEURONS_PER_CELL);

    auto block = cg_mutation::this_thread_block();
    auto laneId = block.thread_rank();

    auto& objects = data.entities.objects;
    auto partition = calcBlockPartition(objects.getNumEntries());

    EntityFactory factory;
    factory.init(&data);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& object = objects.at(index);

        __shared__ Genome* sharedGenome;

        if (laneId == 0) {
            sharedGenome = nullptr;
            if (object->type == ObjectType_Cell) {
                auto& creature = object->typeData.cell.creature;
                int origMutationState = atomicCAS(&creature->mutationState, MutationState_MutationInProgress, MutationState_Mutated);
                if (origMutationState == MutationState_MutationInProgress) {
                    sharedGenome = factory.cloneGenome(creature->genome);
                }
            }
        }
        block.sync();

        if (sharedGenome != nullptr) {

            // Apply mutations to cloned genome
            applyMutations(data, sharedGenome);

            if (laneId == 0) {
                object->typeData.cell.creature->genome = sharedGenome;
            }
        }
        block.sync();
    }
}

__inline__ __device__ void MutationProcessor::applyMutations(SimulationData& data, Genome* genome)
{
    applyMutations_meta(data, genome);
    applyMutations_neurons(data, genome);
    applyMutations_connections(data, genome);
    applyMutations_lineages(data, genome);
}

__inline__ __device__ void MutationProcessor::applyMutations_neurons(SimulationData& data, Genome* genome)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();

    NeuronMutation rates[2] = {genome->neuronMutation1, genome->neuronMutation2};

    for (int rateIndex = 0; rateIndex < 2; ++rateIndex) {
        auto const& rate = rates[rateIndex];
        if (rate.probability <= 0 || (rate.weightSigma <= 0 && rate.biasSigma <= 0 && rate.activationFunctionProbability <= 0)) {
            continue;
        }
        for (int geneIndex = 0; geneIndex < genome->numGenes; ++geneIndex) {
            auto& gene = genome->genes[geneIndex];
            for (int nodeIndex = 0; nodeIndex < gene.numNodes; ++nodeIndex) {
                auto& node = gene.nodes[nodeIndex];
                if (laneId < NEURONS_PER_CELL) {
                    int neuronIndex = laneId;
                    if (data.primaryNumberGen.random() < rate.probability) {
                        if (rate.weightSigma > 0) {
                            for (int weightIndex = 0; weightIndex < NEURONS_PER_CELL; ++weightIndex) {
                                auto& weight = node.neuralNetwork.weights[neuronIndex * NEURONS_PER_CELL + weightIndex];
                                float newValue = weight.getValue() + generateGaussian(data) * rate.weightSigma;
                                newValue = max(-2.0f, min(2.0f, newValue));
                                weight = NeuralNetWeight(newValue);
                            }
                        }
                        if (rate.biasSigma > 0) {
                            auto& bias = node.neuralNetwork.biases[neuronIndex];
                            float newBias = bias + generateGaussian(data) * rate.biasSigma;
                            newBias = max(-2.0f, min(2.0f, newBias));
                            bias = newBias;
                        }
                        if (rate.activationFunctionProbability > 0 && data.primaryNumberGen.random() < rate.activationFunctionProbability) {
                            node.neuralNetwork.activationFunctions[neuronIndex] =
                                static_cast<ActivationFunction>(data.primaryNumberGen.random(ActivationFunction_Count - 1));
                        }
                    }
                }
            }
        }
    }
}

__inline__ __device__ void MutationProcessor::applyMutations_connections(SimulationData& data, Genome* genome)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();

    ConnectionMutation rates[2] = {genome->connectionMutationRate1, genome->connectionMutationRate2};

    for (int rateIndex = 0; rateIndex < 2; ++rateIndex) {
        auto const& rate = rates[rateIndex];
        if (rate.probability <= 0 || rate.sigma <= 0) {
            continue;
        }
        for (int geneIndex = 0; geneIndex < genome->numGenes; ++geneIndex) {
            auto& gene = genome->genes[geneIndex];
            for (int nodeIndex = 0; nodeIndex < gene.numNodes; ++nodeIndex) {
                auto& node = gene.nodes[nodeIndex];
                if (laneId < MAX_OBJECT_CONNECTIONS) {
                    if (data.primaryNumberGen.random() < rate.probability) {
                        auto& weight = node.neuralNetwork.connectionWeights[laneId];
                        float newValue = weight + generateGaussian(data) * rate.sigma;
                        newValue = max(-2.0f, min(2.0f, newValue));
                        weight = newValue;
                    }
                }
            }
        }
    }
}

__inline__ __device__ void MutationProcessor::applyMutations_lineages(SimulationData& data, Genome* genome)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();

    if (laneId == 0 && genome->lineageMutationProbability > 0) {
        if (data.primaryNumberGen.random() < genome->lineageMutationProbability) {
            genome->prevLineageId = genome->lineageId;
            genome->lineageId = data.primaryNumberGen.createLineageId();
        }
    }
}

__inline__ __device__ void MutationProcessor::applyMutations_meta(SimulationData& data, Genome* genome)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();

    if (laneId == 0) {
        // Meta-mutate neuron mutation rates
        float neuronSigma = cudaSimulationParameters.metaMutationNeuronsSigma.value;
        if (neuronSigma > 0) {
            auto mutateFloat = [&](float& val) { val = min(1.0f, max(0.0f, val + generateGaussian(data) * neuronSigma)); };
            mutateFloat(genome->neuronMutation1.probability);
            mutateFloat(genome->neuronMutation1.weightSigma);
            mutateFloat(genome->neuronMutation1.biasSigma);
            mutateFloat(genome->neuronMutation1.activationFunctionProbability);
            mutateFloat(genome->neuronMutation2.probability);
            mutateFloat(genome->neuronMutation2.weightSigma);
            mutateFloat(genome->neuronMutation2.biasSigma);
            mutateFloat(genome->neuronMutation2.activationFunctionProbability);
        }

        // Meta-mutate connection mutation rates
        float connSigma = cudaSimulationParameters.metaMutationConnectionsSigma.value;
        if (connSigma > 0) {
            auto mutateFloat = [&](float& val) { val = min(1.0f, max(0.0f, val + generateGaussian(data) * connSigma)); };
            mutateFloat(genome->connectionMutationRate1.probability);
            mutateFloat(genome->connectionMutationRate1.sigma);
            mutateFloat(genome->connectionMutationRate2.probability);
            mutateFloat(genome->connectionMutationRate2.sigma);
        }

        // Meta-mutate lineage mutation probability
        float lineageSigma = cudaSimulationParameters.metaMutationLineagesSigma.value;
        if (lineageSigma > 0) {
            genome->lineageMutationProbability = min(1.0f, max(0.0f, genome->lineageMutationProbability + generateGaussian(data) * lineageSigma));
        }
    }
}

__inline__ __device__ float MutationProcessor::generateGaussian(SimulationData& data)
{
    // Box-Muller transform for Gaussian random
    float u1 = max(1.0e-7f, data.primaryNumberGen.random());
    float u2 = data.primaryNumberGen.random();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * Const::PI * u2);
}

__inline__ __device__ bool MutationProcessor::isRandomEvent(SimulationData& data, float probability)
{
    if (probability > 0.001f) {
        return data.primaryNumberGen.random() < probability;
    } else {
        return data.primaryNumberGen.random() < probability * 1000 && data.secondaryNumberGen.random() < 0.001f;
    }
}
