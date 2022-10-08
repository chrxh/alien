#pragma once

#include "SimulationData.cuh"
#include "QuantityConverter.cuh"

class NeuralNetProcessor
{
public:
    __inline__ __device__ static void process(Token* token, SimulationData& data);

private:
    __inline__ __device__ static float getWeight(Cell* cell, int index1, int index2);
    __inline__ __device__ static float getInput(Token* token, int index);
    __inline__ __device__ static void setOutput(Token* token, int index, float value);
    __inline__ __device__ static float sigmoid(float z);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void NeuralNetProcessor::process(Token* token, SimulationData& data)
{
    auto cell = token->cell;
    auto partition = calcPartition(8, threadIdx.x, blockDim.x);

    __shared__ float input[8];
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        input[i] = getInput(token, i);
    }
    __syncthreads();

    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        float netInput = 0;
        for (int j = 0; j < 8; ++j) {
            netInput += getWeight(cell, i, j) * input[j];
        }
        setOutput(token, i, sigmoid(netInput));
    }
    __syncthreads();
}
__inline__ __device__ float NeuralNetProcessor::getWeight(Cell* cell, int index1, int index2)
{
    auto flatIndex = index1 + index2 * 8;
    auto data = static_cast<unsigned char>(cell->staticData[flatIndex / 2]);
    int weigthInt = (flatIndex % 2) == 0 ? data & 0xf : (data >> 4) & 0xf;
    return static_cast<float>(weigthInt) / 16 - 0.5f;
}

__inline__ __device__ float NeuralNetProcessor::getInput(Token* token, int index)
{
    int address = Enums::NeuralNet_InOut + index * 2;
    auto byte1 = static_cast<int>(static_cast<unsigned char>(token->memory[address]));
    auto byte2 = static_cast<int>(static_cast<unsigned char>(token->memory[address + 1]));
    return static_cast<float>((byte1 << 8) + byte2) / 65536;
}

__inline__ __device__ void NeuralNetProcessor::setOutput(Token* token, int index, float value)
{
    auto scaledValue = static_cast<int>(value * 65536);
    auto byte1 = scaledValue >> 8;
    auto byte2 = scaledValue & 0xff;

    int address = Enums::NeuralNet_InOut + index * 2;
    token->memory[address] = static_cast<unsigned char>(byte1);
    token->memory[address + 1] = static_cast<unsigned char>(byte2);
}

__inline__ __device__ float NeuralNetProcessor::sigmoid(float z)
{
    return 1.0f / (1.0f + __expf(-z));
}
