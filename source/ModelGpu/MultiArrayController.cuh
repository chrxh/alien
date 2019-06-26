#pragma once

#include "Base.cuh"

template<typename T, int n>
class MultiArrayController
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

    __device__ __inline__ T* getNewElement(int suggestedArrayIndex)
    {
        suggestedArrayIndex = max(min((suggestedArrayIndex - 16) / 48, n - 1), 0);
        auto origSuggestedArrayIndex = suggestedArrayIndex;

        T* result;
        do {
            if (result = _arrays[suggestedArrayIndex].getNewElement()) {
                return result;
            }
            suggestedArrayIndex = (suggestedArrayIndex + n - 1) % n;
        } while (!result && suggestedArrayIndex != origSuggestedArrayIndex);
        return nullptr;
    }

    __device__ __inline__ ArrayController<T>& getArray(int arrayIndex)
    {
        arrayIndex = max(min(arrayIndex, n - 1), 0);
        return _arrays[arrayIndex];
    }

    __device__ __inline__ void reset()
    {
        for (int i = 0; i < n; ++i) {
            _arrays[i].reset();
        }
    }

    __device__ __inline__ void swapArrays(MultiArrayController& other)
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
            int threadsPerBlock = 64/*i * 48 + 16*/; \
            int numBlocks = 64*2/*min(64*64*2/threadsPerBlock, 256)*/; \
            func<<<numBlocks, threadsPerBlock>>>(##__VA_ARGS__, i); \
            cudaDeviceSynchronize(); \
        } \
    }

#define MULTI_CALL_DEBUG(func, ...) for (int i = 0; i < NUM_CLUSTERPOINTERARRAYS; ++i) { \
        auto numEntries = data.entities.clusterPointerArrays.getArray(i).getNumEntries(); \
        if (numEntries > 0) { \
            printf("i: %d, numEntries: %d\n", i, numEntries);\
            int threadsPerBlock = i * 48 + 16; \
            int numBlocks = min(64*64*2/threadsPerBlock, 256); \
            func<<<numBlocks, threadsPerBlock>>>(##__VA_ARGS__, i); \
            cudaDeviceSynchronize(); \
        } \
    }

