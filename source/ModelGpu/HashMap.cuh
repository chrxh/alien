#pragma once

#include "HashSet.cuh"
#include "Array.cuh"

template <typename Key, typename Value, typename Hash = HashFunctor<Key>>
class HashMap
{
public:
    __device__ __inline__ HashMap(int size, ArrayController& arrays)
        : _size(size)
    {
        _entries = arrays.getArray<Entry>(size);
        for (int i = 0; i < size; ++i) {
            _entries[i].free = true;
        }
    }

    __device__ __inline__ void insertOrAssign(Key const& key, Value const& value)
    {
        int index = _hash(key) % _size;
        while (!_entries[index].free) {
            if (_entries[index].key == key) {
                _entries[index].value = value;
                return;
            }
            index = (++index) % _size;
        }
        auto& newEntry = _entries[index];
        newEntry.free = false;
        newEntry.key = key;
        newEntry.value = value;
    }

    __device__ __inline__ bool contains(Key const& key) const
    {
        int index = _hash(key) % _size;
        for (int i = 0; i < _size; ++i, index = (++index) % _size) {
            auto& entry = _entries[index];
            if (entry.free) {
                return false;
            }
            if (entry.key == key) {
                return true;
            }
        }
        return false;
    }

    __device__ __inline__ Value& operator[](Key const& key)
    {
        int index = _hash(key) % _size;
        do {
            auto& entry = _entries[index];
            if (entry.free) {
                entry.free = false;
                entry.key = key;
                return entry.value;
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
        bool free;
        Value value;
        Key key;
    };
    int _size;
    Entry* _entries;
};
