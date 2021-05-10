#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <sstream>

#include "Base/Exceptions.h"

#define KERNEL_CALL(func, ...)  \
        func<<<cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >>>(##__VA_ARGS__); \
        cudaDeviceSynchronize();

#define KERNEL_CALL_1_1(func, ...)  \
        func<<<1, 1>>>(##__VA_ARGS__); \
        cudaDeviceSynchronize();

template< typename T >
void checkAndThrowError(T result, char const *const func, const char *const file, int const line)
{
    if (result) {
        std::stringstream stream;
        stream << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << "("
               << _cudaGetErrorEnum(result) << ") \"" << func << "\"";
        DEVICE_RESET
        throw BugReportException(stream.str().c_str());
    }
}

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define CHECK_FOR_CUDA_ERROR(val) \
    checkAndThrowError( (val), #val, __FILENAME__, __LINE__ )

#define ABORT() *((int*)nullptr) = 1;
