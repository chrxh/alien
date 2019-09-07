#pragma once

#include "HashSet.cuh"
#include "Array.cuh"

template <typename Key, typename Value, typename Hash = HashFunctor<Key>>
class HashMap
{
public:
    __device__ __inline__ void init_blockCall(int size, ArrayController& arrays)
    {
        __shared__ Entry* entries;
        if (0 == threadIdx.x) {
            entries = arrays.getArray<Entry>(size);
        }
        __syncthreads();

        _size = size;
        _entries = entries;

        auto const threadBlock = calcPartition(size, threadIdx.x, blockDim.x);
        for (int i = threadBlock.startIndex; i <= threadBlock.endIndex; ++i) {
            _entries[i].free = 0;
        }
    }

    __device__ __inline__ void insertOrAssign(Key const& key, Value const& value)
    {
        int index = _hash(key) % _size;
        int wasFree;
        do {
            wasFree = atomicExch_block(&_entries[index].free, 1);
            if (1 == wasFree) {
                if (_entries[index].key == key) {
                    _entries[index].value = value;
                    return;
                }
                index = (++index) % _size;
            }
        } while (1 == wasFree);

        auto& newEntry = _entries[index];
        newEntry.key = key;
        newEntry.value = value;
    }

    __device__ __inline__ bool contains(Key const& key) const
    {
        int index = _hash(key) % _size;
        for (int i = 0; i < _size; ++i, index = (++index) % _size) {
            auto& entry = _entries[index];
            if (0 == entry.free) {
                return false;
            }
            if (entry.key == key) {
                return true;
            }
        }
        return false;
    }

    __device__ __inline__ Value at(Key const& key)
    {
        int index = _hash(key) % _size;
        do {
            auto& entry = _entries[index];
            if (0 == entry.free) {
                return Value();
            }
            if (entry.key == key) {
                return entry.value;
            }
            index = (++index) % _size;
        }
        while (true);
    }

private:
    Hash _hash;

    struct Entry
    {
        int free;   //0 = free, 1 = used
        Value value;
        Key key;
    };
    int _size;
    Entry* _entries;
};
