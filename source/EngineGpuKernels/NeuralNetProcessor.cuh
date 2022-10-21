#pragma once

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
    auto partition = calcPartition(MAX_CHANNELS, threadIdx.x, blockDim.x);

    __shared__ Activity outputActivity;
    __shared__ Activity inputActivity;
    if (0 == threadIdx.x) {
        inputActivity = CellFunctionProcessor::calcInputActivity(cell);
    }
    __syncthreads();

    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        float sumInput = 0;
        auto & neuronsState = cell->cellFunctionData.neuronFunction.neuronState;
        for (int j = 0; j < MAX_CHANNELS; ++j) {
            sumInput += neuronsState->weights[i][j] * inputActivity.channels[j];
        }
        sumInput += neuronsState->bias[i];
        outputActivity.channels[i] = sigmoid(sumInput);
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
