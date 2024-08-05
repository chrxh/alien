#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cuda/helper_cuda.h>

#include "Array.cuh"
#include "CudaMemoryManager.cuh"
#include "Base.cuh"
#include "Definitions.cuh"

class CudaNumberGenerator
{
private:
    unsigned int* _currentIndex;
    int* _array;
    int _size;

    unsigned long long int* _currentId;
    unsigned int* _currentSmallId;

public:
    void init(int size)
    {
        _size = size;

        CudaMemoryManager::getInstance().acquireMemory<unsigned int>(1, _currentIndex);
        CudaMemoryManager::getInstance().acquireMemory<int>(size, _array);
        CudaMemoryManager::getInstance().acquireMemory<unsigned long long int>(1, _currentId);
        CudaMemoryManager::getInstance().acquireMemory<unsigned int>(1, _currentSmallId);

        CHECK_FOR_CUDA_ERROR(cudaMemset(_currentIndex, 0, sizeof(unsigned int)));
        unsigned long long int hostCurrentId = 1;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_currentId, &hostCurrentId, sizeof(unsigned long long int), cudaMemcpyHostToDevice));
        unsigned int hostCurrentSmallId = 1;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_currentSmallId, &hostCurrentSmallId, sizeof(unsigned int), cudaMemcpyHostToDevice));

        std::vector<int> randomNumbers(size);
        for (int i = 0; i < size; ++i) {
            randomNumbers[i] = rand();
        }
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_array, randomNumbers.data(), sizeof(int) * size, cudaMemcpyHostToDevice));
    }


    __device__ __inline__ int random(int maxVal)
    {
        int number = getRandomNumber();
        return number % (maxVal + 1);
    }

    __device__ __inline__ float random(float maxVal)
    {
        int number = getRandomNumber();
        return maxVal * static_cast<float>(number) / toFloat(RAND_MAX);
    }

    __device__ __inline__ float random(float minVal, float maxVal)
    {
        int number = getRandomNumber();
        return minVal + (maxVal - minVal) * static_cast<float>(number) / toFloat(RAND_MAX);
    }

    __device__ __inline__ float random()
    {
        int number = getRandomNumber();
        return static_cast<float>(number) / toFloat(RAND_MAX);
    }

    __device__ __inline__ bool randomBool() { return random(1) == 0; }

    __device__ __inline__ uint8_t randomByte() { return static_cast<uint8_t>(random(255)); }

    __device__ __inline__ void randomBytes(uint8_t* data, int size)
    {
        for (int i = 0; i < size; ++i) {
            data[i] = randomByte();
        }
    }

    __device__ __inline__ unsigned long long int createNewId() { return atomicAdd(_currentId, 1); }
    __device__ __inline__ unsigned int createNewSmallId() { return atomicAdd(_currentSmallId, 1); }

    __device__ __inline__ void adaptMaxId(unsigned long long int id) { atomicMax(_currentId, id + 1); }
    __device__ __inline__ void adaptMaxSmallId(unsigned int id) { atomicMax(_currentSmallId, id + 1); }

    void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_currentIndex);
        CudaMemoryManager::getInstance().freeMemory(_array);
        CudaMemoryManager::getInstance().freeMemory(_currentId);
        CudaMemoryManager::getInstance().freeMemory(_currentSmallId);
    }

private:
    __device__ __inline__ int getRandomNumber()
    {
        int index = atomicInc(_currentIndex, _size - 1);
        return _array[index];
    }
};
