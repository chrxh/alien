#pragma once

#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "CellFunctionProcessor.cuh"

class NeuronProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static float applyActivationFunction(NeuronActivationFunction activationFunction, float x);  // maps to [-1, 1]
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void NeuronProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Neuron];
    auto partition = calcPartition(operations.getNumEntries(), blockIdx.x, gridDim.x);
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, statistics, operations.at(i).cell);
    }
}

__inline__ __device__ void NeuronProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    __shared__ Activity outputActivity;
    __shared__ Activity inputActivity;
    if (0 == threadIdx.x) {
        inputActivity = CellFunctionProcessor::calcInputActivity(cell);
        CellFunctionProcessor::updateInvocationState(cell, inputActivity);
    }
    __syncthreads();

    __shared__ float sumInput[MAX_CHANNELS];
    auto channelPartition = calcPartition(MAX_CHANNELS, threadIdx.x, blockDim.x);
    for (int i = channelPartition.startIndex; i <= channelPartition.endIndex; ++i) {
        sumInput[i] = cell->cellFunctionData.neuron.neuronState->biases[i];
    }
    __syncthreads();

    auto matrixPartition = calcPartition(MAX_CHANNELS * MAX_CHANNELS, threadIdx.x, blockDim.x);
    for (int entry = matrixPartition.startIndex; entry <= matrixPartition.endIndex; ++entry) {
        auto& neuronsState = cell->cellFunctionData.neuron.neuronState;

        auto row = entry / MAX_CHANNELS;
        auto col = entry % MAX_CHANNELS;
        atomicAdd(&sumInput[row], neuronsState->weights[entry] * inputActivity.channels[col]);
    }
    __syncthreads();

    for (int i = channelPartition.startIndex; i <= channelPartition.endIndex; ++i) {
        outputActivity.channels[i] = applyActivationFunction(cell->cellFunctionData.neuron.activationFunctions[i], sumInput[i]);  
    }
    __syncthreads();
    

    if (0 == threadIdx.x) {
        CellFunctionProcessor::setActivity(cell, outputActivity);
        CellFunctionProcessor::updateInvocationState(cell, outputActivity);
        statistics.incNumNeuronActivities(cell->color);
    }
    __syncthreads();
}

__inline__ __device__ float NeuronProcessor::applyActivationFunction(NeuronActivationFunction activationFunction, float x)
{
    switch (activationFunction) {
    case NeuronActivationFunction_Sigmoid:
        return 2.0f / (1.0f + __expf(-x)) - 1.0f;
    case NeuronActivationFunction_BinaryStep:
        return x >= NEAR_ZERO ? 1.0f : 0.0f;
    case NeuronActivationFunction_Identity:
        return x;
    case NeuronActivationFunction_Abs:
        return abs(x);
    case NeuronActivationFunction_Gaussian:
        return __expf(-2 * x * x);
    }
    return 0;
}
