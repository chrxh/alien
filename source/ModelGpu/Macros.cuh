#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>


#define KERNEL_CALL(func, ...)  \
        func<<<cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >>>(##__VA_ARGS__); \
        cudaDeviceSynchronize();

#define KERNEL_CALL_1_1(func, ...)  \
        func<<<1, 1>>>(##__VA_ARGS__); \
        cudaDeviceSynchronize();

#include <sstream>
template< typename T >
void checkAndThrowError(T result, char const *const func, const char *const file, int const line)
{
    if (result) {
        std::stringstream stream;
        stream << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result)
            << "(" << _cudaGetErrorEnum(result) << ") \"" << func << "\"";
        DEVICE_RESET
        throw std::exception(stream.str().c_str());
    }
}

#define CHECK_FOR_CUDA_ERROR(val) \
    checkAndThrowError( (val), #val, __FILE__, __LINE__ )
