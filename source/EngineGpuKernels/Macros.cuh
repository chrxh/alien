#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <sstream>


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
               << _cudaGetErrorEnum(result) << ") \"" << func << "\"" << std::endl
               << std::endl
               << "One of the following reasons may be responsible:" << std::endl
               << "- the number of thread blocks may be too high (see 'General settings')" << std::endl
               << "- the array sizes for the entities may be too low (see 'General settings')" << std::endl
               << "- your graphics card has not enough memory" << std::endl
               << std::endl
               << "Please restart the program and try to adjust the setting. If the problem cannot be solved, please "
                  "send the 'log.txt' file to info@alien-project.org.";
        DEVICE_RESET
        throw std::exception(stream.str().c_str());
    }
}

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define CHECK_FOR_CUDA_ERROR(val) \
    checkAndThrowError( (val), #val, __FILENAME__, __LINE__ )
