#pragma once

#include "Definitions.cuh"
#include "HashSet.cuh"

template<typename Key, typename Value>
struct ValueToKeyFunctor
{
};

template <
    typename Key,
    typename Value,
    typename Hash = HashFunctor<Key>,
    typename ValueToKey = ValueToKeyFunctor<Key, Value>>
    class HashMap
{
public:
    __device__ __inline__ HashMap(int size, Value* data)
        : _data(data)
    {
        reset(size);
    }

    __device__ __inline__ void reset(int size)
    {
        _size = size;
        for (int i = 0; i < size; ++i) {
            _data[i] = nullptr;
        }
    }

    __device__ __inline__ void insert(Value const& value)
    {
        auto const key = _valueToKey(value);
        int index = _hash(key) % _size;
        while (_data[index]) {
            if (_valueToKey(_data[index]) == key) {
                return;
            }
            index = (++index) % _size;
        }
        _data[index] = value;
    }

    __device__ __inline__ bool contains(Key const& key)
    {
        int index = _hash(key) % _size;
        for (int i = 0; i < _size; ++i, index = (++index) % _size) {
            if (nullptr == _data[index]) {
                return false;
            }
            else if (_valueToKey(_data[index]) == key) {
                return true;
            }
            
        }
        return false;
    }

    __device__ __inline__ Value& at(Key const& key)
    {
        int index = _hash(key) % _size;
        for (int i = 0; i < _size; ++i, index = (++index) % _size) {
            if (nullptr == _data[index]) {
                return false;
            }
            else if (_valueToKey(_data[index]) == key) {
                return _data[index];
            }

        }
        THROW_NOT_IMPLEMENTED();
    }

private:
    Value* _data;
    int _size;
    Hash _hash;
    ValueToKey _valueToKey;
};
