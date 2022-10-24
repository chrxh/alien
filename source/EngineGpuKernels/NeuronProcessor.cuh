#pragma once

#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "CellFunctionProcessor.cuh"

class NeuronProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationResult& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationResult& result, Cell* cell);

    __inline__ __device__ static float scaledSigmoid(float z);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void NeuronProcessor::process(SimulationData& data, SimulationResult& result)
{
    auto partition = calcPartition(data.cellFunctionOperations[Enums::CellFunction_Neuron].getNumEntries(), blockIdx.x, gridDim.x);
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto cell = data.cellFunctionOperations[Enums::CellFunction_Neuron].at(i).cell;
        processCell(data, result, cell);
    }
}

__inline__ __device__ void NeuronProcessor::processCell(SimulationData& data, SimulationResult& result, Cell* cell)
{
    __shared__ Activity outputActivity;
    __shared__ Activity inputActivity;
    if (0 == threadIdx.x) {
        inputActivity = CellFunctionProcessor::calcInputActivity(cell);
    }
    __syncthreads();

    __shared__ float sumInput[MAX_CHANNELS];
    auto channelPartition = calcPartition(MAX_CHANNELS, threadIdx.x, blockDim.x);
    for (int i = channelPartition.startIndex; i <= channelPartition.endIndex; ++i) {
        sumInput[i] = cell->cellFunctionData.neuron.neuronState->bias[i];
    }
    __syncthreads();

    auto matrixPartition = calcPartition(MAX_CHANNELS * MAX_CHANNELS, threadIdx.x, blockDim.x);
    for (int c = matrixPartition.startIndex; c <= matrixPartition.endIndex; ++c) {
        auto& neuronsState = cell->cellFunctionData.neuron.neuronState;

        auto row = c / MAX_CHANNELS;
        auto col = c % MAX_CHANNELS;
        atomicAdd_block(&sumInput[row], neuronsState->weights[col + row * MAX_CHANNELS] * inputActivity.channels[col]);
    }
    __syncthreads();

    for (int i = channelPartition.startIndex; i <= channelPartition.endIndex; ++i) {
        outputActivity.channels[i] = scaledSigmoid(sumInput[i]);  
    }
    __syncthreads();
    

    if (0 == threadIdx.x) {
        CellFunctionProcessor::setActivity(cell, outputActivity);
    }
    __syncthreads();
}

// maps to [-1, 1]
__inline__ __device__ float NeuronProcessor::scaledSigmoid(float z)
{
    return 2.0f / (1.0f + __expf(-z)) - 1.0f;
}
