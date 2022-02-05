#pragma once

#include "HashSet.cuh"
#include "Array.cuh"

template <typename Key, typename Value, typename Hash = HashFunctor<Key>>
class HashMap
{
public:
    __device__ __inline__ void init_block(int size, RawMemory& arrays)
    {
        __shared__ Entry* entries;
        if (0 == threadIdx.x) {
            entries = arrays.getArray<Entry>(size);
        }
        __syncthreads();

        _size = size;
        _entries = entries;
    }

    __device__ __inline__ void reset_block()
    {
        auto const threadBlock = calcPartition(_size, threadIdx.x, blockDim.x);
        for (int i = threadBlock.startIndex; i <= threadBlock.endIndex; ++i) {
            _entries[i].setFree(0);
            _entries[i].initLock();
        }
        __syncthreads();
    }

    //return true if key was present
    __device__ __inline__ bool insertOrAssign(Key const& key, Value const& value)
    {
        int index = _hash(key) % _size;
        int wasFree;
        do {
            auto& entry = _entries[index];
            entry.getLock(1);
            wasFree = entry.setFree(1);
            if (1 == wasFree) {
                if (entry.getKey() == key) {
                    entry.setValue(value);
                    entry.releaseLock();
                    return true;
                }
                entry.releaseLock();
                index = (++index) % _size;
            }
        } while (1 == wasFree);

        auto& newEntry = _entries[index];
        newEntry.setKey(key);
        newEntry.setValue(value);
        newEntry.releaseLock();
        return false;
    }

    __device__ __inline__ bool contains(Key const& key) const
    {
        int index = _hash(key) % _size;
        for (int i = 0; i < 10; ++i, index = (++index) % _size) {   //workaround: 10 is set to avoid too long runtime
            auto& entry = _entries[index];
            entry.getLock(2);
            if (0 == entry.getFree()) {
                entry.releaseLock();
                return false;
            }
            if (entry.getKey() == key) {
                entry.releaseLock();
                return true;
            }
            entry.releaseLock();
        }
        return false;
    }

    __device__ __inline__ Value at(Key const& key)
    {
        for (int i = 0; i < 10; ++i) {  //workaround: 10 is set to avoid too long runtime
            int index = _hash(key) % _size;
            auto& entry = _entries[index];
            entry.getLock(3);
            if (0 == entry.getFree()) {
                entry.releaseLock();
                return Value();
            }
            if (entry.getKey() == key) {
                auto result = entry.getValue();
                entry.releaseLock();
                return result;
            }
            entry.releaseLock();
            index = (++index) % _size;
        }
        return Value();
    }

private:
    Hash _hash;

    class Entry
    {
    public:
        __device__ __inline__ int setFree(int value)
        {
            int origValue = _free;
            _free = value;
            return origValue;
        }
        __device__ __inline__ int getFree()
        {
            return _free;
        }

        __device__ __inline__ void setValue(Value const& value)
        {
            _value = value;
        }
        __device__ __inline__ Value getValue()
        {
            return _value;
        }

        __device__ __inline__ void setKey(Key const& value)
        {
            _key = value;
        }
        __device__ __inline__ Key getKey()
        {
            return _key;
        }

        __device__ __inline__ void initLock()
        {
            atomicExch_block(&_locked, 0);
        }

        __device__ __inline__ bool tryLock()
        {
            return 0 == atomicExch_block(&_locked, 1);
        }

        __device__ __inline__ void getLock(int parameter)
        {
            while (1 == atomicExch_block(&_locked, 1)) {}
            __threadfence_block();
        }

        __device__ __inline__ void releaseLock()
        {
            __threadfence_block();
            atomicExch_block(&_locked, 0);
        }

    private:
        int _free;   //0 = free, 1 = used
        int _locked;	//0 = unlocked, 1 = locked
        Value _value;
        Key _key;
    };
    int _size;
    Entry* _entries;
};
