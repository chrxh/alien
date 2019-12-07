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

    __device__ __inline__ Array<T>& getArray(int arrayIndex)
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
    Array<T> _arrays; //using 1 array currently
};

