#pragma once

#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "CellFunctionProcessor.cuh"

class NeuralNetProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationResult& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationResult& result, Cell* cell);

    __inline__ __device__ static float sigmoid(float z);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void NeuralNetProcessor::process(SimulationData& data, SimulationResult& result)
{
    auto partition = calcPartition(data.cellFunctionOperations[Enums::CellFunction_Neurons].getNumEntries(), blockIdx.x, gridDim.x);
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto cell = data.cellFunctionOperations[Enums::CellFunction_Neurons].at(i).cell;
        processCell(data, result, cell);
    }
}

__inline__ __device__ void NeuralNetProcessor::processCell(SimulationData& data, SimulationResult& result, Cell* cell)
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
        sumInput[i] = cell->cellFunctionData.neuronFunction.neuronState->bias[i];
    }
    __syncthreads();

    auto matrixPartition = calcPartition(MAX_CHANNELS * MAX_CHANNELS, threadIdx.x, blockDim.x);
    for (int c = matrixPartition.startIndex; c <= matrixPartition.endIndex; ++c) {
        auto& neuronsState = cell->cellFunctionData.neuronFunction.neuronState;

        auto i = c / MAX_CHANNELS;
        auto j = c % MAX_CHANNELS;
        atomicAdd_block(&sumInput[i], neuronsState->weights[i][j] * inputActivity.channels[j]);
    }
    __syncthreads();

    for (int i = channelPartition.startIndex; i <= channelPartition.endIndex; ++i) {
        outputActivity.channels[i] = sigmoid(sumInput[i]);
    }
    __syncthreads();
    

    if (0 == threadIdx.x) {
        CellFunctionProcessor::setActivity(cell, outputActivity);
    }
    __syncthreads();
}

__inline__ __device__ float NeuralNetProcessor::sigmoid(float z)
{
    return 1.0f / (1.0f + __expf(-z));
}
