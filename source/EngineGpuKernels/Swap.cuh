#pragma once

template <typename T>
__host__ __device__ __inline__ void swap(T& a, T& b)
{
    T temp = a;
    a = b;
    b = temp;
}
