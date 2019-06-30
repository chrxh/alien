#pragma once

#include "Base.cuh"

template<typename T>
class ClusterPointerMultiArrayController
{
public:
    void init(int numArrays, int* sizes)
    {
        _numArrays = numArrays;
        for (int i = 0; i < _numArrays; ++i) {
            _arrays/*[i]*/.init(sizes[i]);
        }
    }

    void free()
    {
        for (int i = 0; i < _numArrays; ++i) {
            _arrays/*[i]*/.free();
        }
    }

    __device__ __inline__ T* getNewClusterPointer(int clusterSize)
    {
        auto index = max(min((clusterSize - 16) / 48, _numArrays - 1), 0);
        auto origIndex = index;

        T* result;
        do {
            if (result = _arrays/*[index]*/.getNewElement()) {
                return result;
            }
            index = (index + _numArrays - 1) % _numArrays;
        } while (!result && index != origIndex);
        return nullptr;
    }

    __device__ __inline__ ArrayController<T>& getArray(int arrayIndex)
    {
        return _arrays/*[arrayIndex]*/;
    }

    __device__ __inline__ void reset()
    {
        for (int i = 0; i < _numArrays; ++i) {
            _arrays/*[i]*/.reset();
        }
    }

    __device__ __inline__ void swapArrays(ClusterPointerMultiArrayController& other)
    {
        for (int i = 0; i < _numArrays; ++i) {
            _arrays/*[i]*/.swapArray(other._arrays/*[i]*/);
        }
    }

private:
    int _numArrays = 0;
    ArrayController<T> _arrays; //using 1 array currently
};

#define MULTI_CALL(func, ...) for (int i = 0; i < cudaConstants.NUM_CLUSTERPOINTERARRAYS; ++i) { \
        auto numEntries = data.entities.clusterPointerArrays.getArray(i).getNumEntries(); \
        if (numEntries > 0) { \
            int threadsPerBlock = cudaConstants.NUM_THREADS_PER_BLOCK/*i * 48 + 16*/; \
            int numBlocks = cudaConstants.NUM_BLOCKS /*min(64*64*2/threadsPerBlock, 256)*/; \
            func<<<numBlocks, threadsPerBlock>>>(##__VA_ARGS__, i); \
        } \
    } \
            cudaDeviceSynchronize();

#define MULTI_CALL_DEBUG(func, ...) for (int i = 0; i < cudaConstants.NUM_CLUSTERPOINTERARRAYS; ++i) { \
        auto numEntries = data.entities.clusterPointerArrays.getArray(i).getNumEntries(); \
        if (numEntries > 0) { \
            printf("i: %d, numEntries: %d\n", i, numEntries);\
            int threadsPerBlock = cudaConstants.NUM_THREADS_PER_BLOCK/*i * 48 + 16*/; \
            int numBlocks = cudaConstants.NUM_BLOCKS /*min(64*64*2/threadsPerBlock, 256)*/; \
            func<<<numBlocks, threadsPerBlock>>>(##__VA_ARGS__, i); \
        } \
    } \
            cudaDeviceSynchronize();

