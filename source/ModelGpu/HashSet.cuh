#pragma once

template<typename T>
struct HashFunctor
{
};


template<typename T>
struct HashFunctor<T*>
{
    __device__ __inline__ int operator()(T* const& element)
    {
        return abs(static_cast<int>(reinterpret_cast<std::uintptr_t>(element)) * 17);
    }
};

template<typename T, typename Hash = HashFunctor<T>>
class HashSet
{
public:
    __device__ __inline__ HashSet(int size, T* data)
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

    __device__ __inline__ void insert(T const& element)
    {
        int index = _hash(element) % _size;
        while (_data[index]) {
            if (_data[index] == element) {
                return;
            }
            index = (++index) % _size;
        }
        _data[index] = element;
    }

    __device__ __inline__ bool contains(T const& element)
    {
        int index = _hash(element) % _size;
        for (int i = 0; i < _size; ++i, index = (++index) % _size) {
            if (_data[index] == element) {
                return true;
            }
            else if (nullptr == _data[index]) {
                return false;
            }
        }
        return false;
    }

private:
    T* _data;
    int _size;
    Hash _hash;
};

