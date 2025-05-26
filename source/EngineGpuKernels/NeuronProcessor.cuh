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

    __inline__ __device__ static float applyActivationFunction(ActivationFunction activationFunction, float x);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void NeuronProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& cells = data.objects.cells;
    auto partition = calcPartition(cells.getNumEntries(), blockIdx.x, gridDim.x);

    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto& cell = cells.at(i);
        if (cell->neuralNetwork) {
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
    if (threadIdx.x < MAX_CHANNELS) {
        sumInput[threadIdx.x] = cell->neuralNetwork->biases[threadIdx.x];
    }
    __syncthreads();

    auto& neuronsState = cell->neuralNetwork;

    auto row = threadIdx.x / MAX_CHANNELS;
    auto col = threadIdx.x % MAX_CHANNELS;
    atomicAdd_block(&sumInput[row], neuronsState->weights[threadIdx.x] * signal.channels[col]);

    __syncthreads();

    if (threadIdx.x < MAX_CHANNELS) {
        signal.channels[threadIdx.x] = max(
            -2.0f,
            min(2.0f,
                applyActivationFunction(cell->neuralNetwork->activationFunctions[threadIdx.x], sumInput[threadIdx.x])));  // truncate value to avoid overflow
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
        return x;
    case ActivationFunction_Abs:
        return abs(x);
    case ActivationFunction_Gaussian:
        return __expf(-2 * x * x);
    }
    return 0;
}
