#pragma once

#include "RawMemory.cuh"

template <typename Value>
class List
{
public:
    __device__ __inline__ void init()
    {
        _startListEntry = nullptr;
        _endListEntry = nullptr;
        _numElements = 0;
    }

    __device__ __inline__ void pushBack(Value const& value, RawMemory* dynamicMemory)
    {
        atomicAdd(&_numElements, 1);
        auto newEntry = dynamicMemory->getArray<ListEntry>(1);
        newEntry->value = value;
        newEntry->nextListEntry = nullptr;

        auto origEndListEntry = atomicExch<ListEntry>(&_endListEntry, newEntry);

        __threadfence();

        if (origEndListEntry) {
            origEndListEntry->nextListEntry = newEntry;
        }
        else {
            _startListEntry = newEntry;
        }
    }

    __device__ __inline__ Value* asArray(RawMemory* dynamicMemory) const
    {
        Value* result = dynamicMemory->getArray<Value>(_numElements);
        auto index = 0;
        auto entry = _startListEntry;
        while (entry) {
            result[index++] = entry->value;
            entry = entry->nextListEntry;
        }
        return result;
    }

    __device__ __inline__ int getSize() const
    {
        return _numElements;
    }

private:
    struct ListEntry {
        Value value;
        ListEntry* nextListEntry;
    };

    ListEntry* _startListEntry;
    ListEntry* _endListEntry;
    int _numElements;
};