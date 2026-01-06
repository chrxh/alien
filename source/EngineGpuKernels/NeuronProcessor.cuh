#pragma once

#include <cooperative_groups.h>

#include "sm_60_atomic_functions.h"

#include "SignalProcessor.cuh"
#include "SimulationData.cuh"

namespace cg = cooperative_groups;

class NeuronProcessor
{
public:
    // needs to be called with MAX_CHANNELS * MAX_CHANNELS threads
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

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
    auto partition = calcBlockPartition(cells.getNumEntries());

    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto& cell = cells.at(i);
        if (cell->neuralNetwork) {
            processCell(data, statistics, cell);
        }
    }
}

__inline__ __device__ void NeuronProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto block = cg::this_thread_block();

    __shared__ Signal signal;
    __shared__ SignalState signalState;
    if (block.thread_rank() == 0) {
        signal = cell->signal;
        signalState = cell->signalState;
    }
    block.sync();

    if (signalState != SignalState_Active) {
        return;
    }

    __shared__ float sumInput[MAX_CHANNELS];
    if (block.thread_rank() < MAX_CHANNELS) {
        sumInput[block.thread_rank()] = cell->neuralNetwork->biases[block.thread_rank()];
    }
    block.sync();

    auto& neuronsState = cell->neuralNetwork;

    auto row = block.thread_rank() / MAX_CHANNELS;
    auto col = block.thread_rank() % MAX_CHANNELS;
    atomicAdd_block(&sumInput[row], neuronsState->weights[block.thread_rank()] * signal.channels[col]);

    block.sync();

    if (block.thread_rank() < MAX_CHANNELS) {
        signal.channels[block.thread_rank()] = max(
            -2.0f,
            min(2.0f,
                applyActivationFunction(
                    cell->neuralNetwork->activationFunctions[block.thread_rank()], sumInput[block.thread_rank()])));  // truncate value to avoid overflow
    }
    block.sync();

    if (block.thread_rank() == 0) {
        cell->signal = signal;
        statistics.incNumNeuronActivities(cell->color);
    }
    block.sync();
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
