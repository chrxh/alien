#pragma once

#include "Base.cuh"

template<typename T, int n>
class ClusterPointerMultiArrayController
{
public:
    void init(int sizes[n])
    {
        for (int i = 0; i < n; ++i) {
            _arrays[i].init(sizes[i]);
        }
    }

    void free()
    {
        for (int i = 0; i < n; ++i) {
            _arrays[i].free();
        }
    }

    __device__ __inline__ T* getNewClusterPointer(int clusterSize)
    {
        auto index = max(min((clusterSize - 16) / 48, n - 1), 0);
        auto origIndex = index;

        T* result;
        do {
            if (result = _arrays[index].getNewElement()) {
                return result;
            }
            index = (index + n - 1) % n;
        } while (!result && index != origIndex);
        return nullptr;
    }

    __device__ __inline__ ArrayController<T>& getArray(int arrayIndex)
    {
        return _arrays[arrayIndex];
    }

    __device__ __inline__ void reset()
    {
        for (int i = 0; i < n; ++i) {
            _arrays[i].reset();
        }
    }

    __device__ __inline__ void swapArrays(ClusterPointerMultiArrayController& other)
    {
        for (int i = 0; i < n; ++i) {
            _arrays[i].swapArray(other._arrays[i]);
        }
    }

private:
    ArrayController<T> _arrays[n];
};

#define MULTI_CALL(func, ...) for (int i = 0; i < NUM_CLUSTERPOINTERARRAYS; ++i) { \
        auto numEntries = data.entities.clusterPointerArrays.getArray(i).getNumEntries(); \
        if (numEntries > 0) { \
            int threadsPerBlock = NUM_THREADS_PER_BLOCK/*i * 48 + 16*/; \
            int numBlocks = NUM_BLOCKS /*min(64*64*2/threadsPerBlock, 256)*/; \
            func<<<numBlocks, threadsPerBlock>>>(##__VA_ARGS__, i); \
        } \
    } \
            cudaDeviceSynchronize();

#define MULTI_CALL_DEBUG(func, ...) for (int i = 0; i < NUM_CLUSTERPOINTERARRAYS; ++i) { \
        auto numEntries = data.entities.clusterPointerArrays.getArray(i).getNumEntries(); \
        if (numEntries > 0) { \
            printf("i: %d, numEntries: %d\n", i, numEntries);\
            int threadsPerBlock = NUM_THREADS_PER_BLOCK/*i * 48 + 16*/; \
            int numBlocks = NUM_BLOCKS /*min(64*64*2/threadsPerBlock, 256)*/; \
            func<<<numBlocks, threadsPerBlock>>>(##__VA_ARGS__, i); \
        } \
    } \
            cudaDeviceSynchronize();

