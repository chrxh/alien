#pragma once

#include "Array.cuh"
#include "Base.cuh"

// Open-addressing hash map from entity ids to entities on the device. It is rebuilt from scratch whenever the entities are moved.
template <typename T>
class IdMap
{
public:
    static uint64_t constexpr EmptyKey = 0xffffffffffffffffull;

    __host__ void init()
    {
        _keys.init();
        _values.init();
    }

    __host__ void free()
    {
        _keys.free();
        _values.free();
    }

    __host__ uint64_t getCapacity_host() const { return _keys.getCapacity_host(); }

    // The capacity must be a power of two
    __host__ void resize(uint64_t capacity) const
    {
        _keys.resize(capacity);
        _values.resize(capacity);
    }

    __device__ __inline__ void clear_system() const
    {
        auto keys = _keys.getArray();
        auto const partition = calcSystemThreadPartition(_keys.getCapacity());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            keys[index] = EmptyKey;
        }
    }

    // Returns false if the map is full
    __device__ __inline__ bool insert(uint64_t id, T* value) const
    {
        auto keys = _keys.getArray();
        auto values = _values.getArray();
        auto capacity = _keys.getCapacity();
        auto mask = capacity - 1;
        auto index = hash(id) & mask;
        uint64_t const emptyKey = EmptyKey;
        for (uint64_t probe = 0; probe < capacity; ++probe) {
            auto origKey = alienAtomicCAS64(&keys[index], emptyKey, id);
            if (origKey == emptyKey || origKey == id) {
                values[index] = value;
                return true;
            }
            index = (index + 1) & mask;
        }
        return false;
    }

    __device__ __inline__ T* find(uint64_t id) const
    {
        auto keys = _keys.getArray();
        auto values = _values.getArray();
        auto capacity = _keys.getCapacity();
        auto mask = capacity - 1;
        auto index = hash(id) & mask;
        for (uint64_t probe = 0; probe < capacity; ++probe) {
            auto key = keys[index];
            if (key == id) {
                return values[index];
            }
            if (key == EmptyKey) {
                return nullptr;
            }
            index = (index + 1) & mask;
        }
        return nullptr;
    }

private:
    __device__ __inline__ static uint64_t hash(uint64_t id)
    {
        id += 0x9e3779b97f4a7c15ull;
        id = (id ^ (id >> 30)) * 0xbf58476d1ce4e5b9ull;
        id = (id ^ (id >> 27)) * 0x94d049bb133111ebull;
        return id ^ (id >> 31);
    }

    Array<uint64_t> _keys;
    Array<T*> _values;
};
