#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cuda/helper_cuda.h>

#include "EngineInterface/Ids.h"

#include "Array.cuh"
#include "CudaMemoryManager.cuh"
#include "Base.cuh"
#include "Definitions.cuh"

class CudaNumberGenerator
{
public:

    // Methods for host
    __host__ void init(int size)
    {
        _size = size;

        CudaMemoryManager::getInstance().acquireMemory(1, _currentRandomNumberIndex);
        CudaMemoryManager::getInstance().acquireMemory(size, _array);
        CudaMemoryManager::getInstance().acquireMemory(size, _ids);

        CHECK_FOR_CUDA_ERROR(cudaMemset(_currentRandomNumberIndex, 0, sizeof(uint32_t)));
        std::vector<int> randomNumbers(size);
        for (int i = 0; i < size; ++i) {
            randomNumbers[i] = rand();
        }
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_array, randomNumbers.data(), sizeof(int) * size, cudaMemcpyHostToDevice));

        Ids hostIds;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_ids, &hostIds, sizeof(hostIds), cudaMemcpyHostToDevice));
    }

    __host__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_currentRandomNumberIndex);
        CudaMemoryManager::getInstance().freeMemory(_array);
        CudaMemoryManager::getInstance().freeMemory(_ids);
    }

    __host__ __inline__ Ids getIds_host()
    {
        Ids hostIds;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&hostIds, _ids, sizeof(hostIds), cudaMemcpyDeviceToHost));
        return hostIds;
    }

    // Methods for device
    __device__ __inline__ int random(int maxVal)
    {
        int number = getRandomNumber();
        return number % (maxVal + 1);
    }

    __device__ __inline__ float random(float maxVal)
    {
        int number = getRandomNumber();
        return maxVal * static_cast<float>(number) / RAND_MAX;
    }

    __device__ __inline__ float random(float minVal, float maxVal)
    {
        int number = getRandomNumber();
        return minVal + (maxVal - minVal) * static_cast<float>(number) / RAND_MAX;
    }

    __device__ __inline__ float random()
    {
        int number = getRandomNumber();
        return static_cast<float>(number) / RAND_MAX;
    }

    __device__ __inline__ bool randomBool() { return random(1) == 0; }

    __device__ __inline__ uint8_t randomByte() { return static_cast<uint8_t>(random(255)); }

    __device__ __inline__ void randomBytes(uint8_t* data, int size)
    {
        for (int i = 0; i < size; ++i) {
            data[i] = randomByte();
        }
    }

    __device__ __inline__ uint64_t createObjectId() { return atomicAdd(&_ids->currentObjectId, 1); }
    __device__ __inline__ uint64_t createCreatureId() { return atomicAdd(&_ids->currentCreatureId, 1); }
    __device__ __inline__ uint32_t createMutationId() { return atomicAdd(&_ids->currentMutationId, 1); }

    __device__ __inline__ void adaptMaxIds(Ids const& ids)
    {
        atomicMax(&_ids->currentObjectId, ids.currentObjectId + 1);
        atomicMax(&_ids->currentCreatureId, ids.currentCreatureId + 1);
        atomicMax(&_ids->currentMutationId, ids.currentMutationId + 1);
    }

private:
    __device__ __inline__ int getRandomNumber()
    {
        int index = atomicInc(_currentRandomNumberIndex, _size - 1);
        return _array[index];
    }

    uint32_t* _currentRandomNumberIndex = nullptr;
    int* _array = nullptr;
    int _size = 0;

    Ids* _ids = nullptr;
};
