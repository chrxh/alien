#pragma once

#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "SignalProcessor.cuh"

class NeuronProcessor
{
public:
    // needs to be called with MAX_CHANNELS * MAX_CHANNELS threads
    __inline__ __device__ static void process(
        SimulationData& data,
        SimulationStatistics& statistics);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static float applyActivationFunction(ActivationFunction activationFunction, float x);  // maps to [-1, 1]
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void NeuronProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcPartition(cells.getNumEntries(), blockIdx.x, gridDim.x);

    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto& cell = cells.at(i);
        if (cell->cellType != CellType_Structure && cell->cellType != CellType_Free) {
            processCell(data, statistics, cell);
        }
    }
}

__inline__ __device__ void NeuronProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    __shared__ Signal signal;
    if (0 == threadIdx.x) {
        signal = cell->signal;
    }
    __syncthreads();

    if (!signal.active) {
        return;
    }

    __shared__ float sumInput[MAX_CHANNELS];
    auto channelPartition = calcPartition(MAX_CHANNELS, threadIdx.x, blockDim.x);
    for (int i = channelPartition.startIndex; i <= channelPartition.endIndex; ++i) {
        sumInput[i] = cell->neuralNetwork->biases[i];
    }
    __syncthreads();

    auto& neuronsState = cell->neuralNetwork;

    auto row = threadIdx.x / MAX_CHANNELS;
    auto col = threadIdx.x % MAX_CHANNELS;
    atomicAdd(&sumInput[row], neuronsState->weights[threadIdx.x] * signal.channels[col]);

    __syncthreads();

    for (int i = channelPartition.startIndex; i <= channelPartition.endIndex; ++i) {
        signal.channels[i] = applyActivationFunction(cell->neuralNetwork->activationFunctions[i], sumInput[i]);  
    }
    __syncthreads();
    

    if (0 == threadIdx.x) {
        cell->signal = signal;
        statistics.incNumNeuronActivities(cell->color);
    }
    __syncthreads();
}

__inline__ __device__ float NeuronProcessor::applyActivationFunction(ActivationFunction activationFunction, float x)
{
    switch (activationFunction) {
    case ActivationFunction_Sigmoid:
        return 2.0f / (1.0f + __expf(-x)) - 1.0f;
    case ActivationFunction_BinaryStep:
        return x >= NEAR_ZERO ? 1.0f : 0.0f;
    case ActivationFunction_Identity:
        return max(-1.0f, min(1.0f, x));
    case ActivationFunction_Abs:
        return min(1.0f, abs(x));
    case ActivationFunction_Gaussian:
        return __expf(-2 * x * x);
    }
    return 0;
}
