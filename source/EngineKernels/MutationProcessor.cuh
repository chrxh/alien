#pragma once

#include <cooperative_groups.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/NeuralNetWeight.h>
#include <EngineInterface/SimulationParameters.h>

#include "EntityFactory.cuh"
#include "SimulationStatistics.cuh"

namespace cg_mutation = cooperative_groups;

class MutationProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);
    __inline__ __device__ static void applyMutations(SimulationData& data, Genome* genome);

private:
    __inline__ __device__ static void applyMutations_neurons(SimulationData& data, Genome* genome, float& accumulatedMutations);
    __inline__ __device__ static void applyMutations_connections(SimulationData& data, Genome* genome, float& accumulatedMutations);
    __inline__ __device__ static void applyMutations_meta(SimulationData& data, Genome* genome);
    __inline__ __device__ static void updateAccumulatedMutationsAndLineageId(SimulationData& data, Genome* genome, float& accumulatedMutations);
    __inline__ __device__ static float generateGaussian(SimulationData& data);
    __inline__ __device__ static bool isRandomEvent(SimulationData& data, float probability);

    __inline__ __device__ static int getNumberOfNodes(Genome* genome);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void MutationProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    DEVICE_CHECK(blockDim.x == NEURONS_PER_CELL);

    auto block = cg_mutation::this_thread_block();
    auto laneId = block.thread_rank();

    auto& objects = data.entities.objects;
    auto partition = calcBlockPartition(objects.getNumEntries());

    EntityFactory factory;
    factory.init(&data);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& object = objects.at(index);

        __shared__ bool mutationNeeded;
        if (laneId == 0) {
            mutationNeeded = data.primaryNumberGen.random() < 0.1f;
        }
        block.sync();
        if (mutationNeeded == false) {
            continue;
        }

        __shared__ Genome* clonedGenome;

        if (laneId == 0) {
            clonedGenome = nullptr;
            if (object->type == ObjectType_Cell) {
                auto& creature = object->typeData.cell.creature;
                int origMutationState = atomicCAS(&creature->mutationState, MutationState_MutationInProgress, MutationState_Mutated);
                if (origMutationState == MutationState_MutationInProgress) {
                    clonedGenome = factory.cloneGenome(creature->genome);
                }
            }
        }
        block.sync();

        if (clonedGenome != nullptr) {

            // Apply mutations to cloned genome
            applyMutations(data, clonedGenome);

            if (laneId == 0) {
                object->typeData.cell.creature->genome = clonedGenome;
            }
        }
        block.sync();
    }
}

__inline__ __device__ void MutationProcessor::applyMutations(SimulationData& data, Genome* genome)
{
    __shared__ float accumulatedMutations;
    auto block = cg_mutation::this_thread_block();
    auto laneId = block.thread_rank();

    if (laneId == 0) {
        accumulatedMutations = 0.0f;
    }
    block.sync();

    applyMutations_meta(data, genome);
    applyMutations_neurons(data, genome, accumulatedMutations);
    applyMutations_connections(data, genome, accumulatedMutations);

    updateAccumulatedMutationsAndLineageId(data, genome, accumulatedMutations);
}

__inline__ __device__ void MutationProcessor::applyMutations_neurons(SimulationData& data, Genome* genome, float& accumulatedMutations)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();

    NeuronMutation rates[2] = {genome->mutationRates.neuronMutation1, genome->mutationRates.neuronMutation2};

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
                                auto delta = generateGaussian(data) * rate.weightSigma;
                                float newValue = weight.getValue() + delta;
                                newValue = max(-2.0f, min(2.0f, newValue));
                                weight = NeuralNetWeight(newValue);
                                atomicAdd_block(&accumulatedMutations, abs(delta));
                            }
                        }
                        if (rate.biasSigma > 0) {
                            auto& bias = node.neuralNetwork.biases[neuronIndex];
                            auto delta = generateGaussian(data) * rate.biasSigma;
                            float newBias = bias + delta;
                            newBias = max(-2.0f, min(2.0f, newBias));
                            bias = newBias;
                            atomicAdd_block(&accumulatedMutations, abs(delta));
                        }
                        if (rate.activationFunctionProbability > 0 && data.primaryNumberGen.random() < rate.activationFunctionProbability) {
                            node.neuralNetwork.activationFunctions[neuronIndex] =
                                static_cast<ActivationFunction>(data.primaryNumberGen.random(ActivationFunction_Count - 1));
                            atomicAdd_block(&accumulatedMutations, 1.0f);
                        }
                    }
                }
            }
        }
    }
}

__inline__ __device__ void MutationProcessor::applyMutations_connections(SimulationData& data, Genome* genome, float& accumulatedMutations)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();

    ConnectionMutation rates[2] = {genome->mutationRates.connectionMutation1, genome->mutationRates.connectionMutation2};

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
                        auto delta = generateGaussian(data) * rate.sigma;
                        float newValue = weight + delta;
                        newValue = max(-2.0f, min(2.0f, newValue));
                        weight = newValue;
                        atomicAdd_block(&accumulatedMutations, abs(delta));
                    }
                }
            }
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
            mutateFloat(genome->mutationRates.neuronMutation1.probability);
            mutateFloat(genome->mutationRates.neuronMutation1.weightSigma);
            mutateFloat(genome->mutationRates.neuronMutation1.biasSigma);
            mutateFloat(genome->mutationRates.neuronMutation1.activationFunctionProbability);
            mutateFloat(genome->mutationRates.neuronMutation2.probability);
            mutateFloat(genome->mutationRates.neuronMutation2.weightSigma);
            mutateFloat(genome->mutationRates.neuronMutation2.biasSigma);
            mutateFloat(genome->mutationRates.neuronMutation2.activationFunctionProbability);
        }

        // Meta-mutate connection mutation rates
        float connSigma = cudaSimulationParameters.metaMutationConnectionsSigma.value;
        if (connSigma > 0) {
            auto mutateFloat = [&](float& val) { val = min(1.0f, max(0.0f, val + generateGaussian(data) * connSigma)); };
            mutateFloat(genome->mutationRates.connectionMutation1.probability);
            mutateFloat(genome->mutationRates.connectionMutation1.sigma);
            mutateFloat(genome->mutationRates.connectionMutation2.probability);
            mutateFloat(genome->mutationRates.connectionMutation2.sigma);
        }
    }
}

__inline__ __device__ void MutationProcessor::updateAccumulatedMutationsAndLineageId(SimulationData& data, Genome* genome, float& accumulatedMutations)
{
    auto laneId = cg_mutation::this_thread_block().thread_rank();
    if (laneId == 0) {
        auto numNodes = getNumberOfNodes(genome);
        if (numNodes == 0) {
            return;
        }
        genome->accumulatedMutations += accumulatedMutations / toFloat(numNodes);
        if (genome->accumulatedMutations > cudaSimulationParameters.accumulatedMutationsForNewLineage.value) {
            genome->prevLineageId = genome->lineageId;
            genome->lineageId = data.primaryNumberGen.createLineageId();
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

__inline__ __device__ int MutationProcessor::getNumberOfNodes(Genome* genome)
{
    auto totalNodes = 0;
    for (int geneIndex = 0; geneIndex < genome->numGenes; ++geneIndex) {
        totalNodes += genome->genes[geneIndex].numNodes;
    }
    return totalNodes;
}
