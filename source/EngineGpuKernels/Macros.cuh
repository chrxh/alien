#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/helper_cuda.h>
#include <sstream>

#include "Base/Exceptions.h"

#define KERNEL_CALL_HOST(func, ...) \
    func<<<1, 1>>>(__VA_ARGS__); \
    cudaDeviceSynchronize(); \
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());

#define KERNEL_CALL(func, ...)  \
        func<<<gpuConstants.NUM_BLOCKS, gpuConstants.NUM_THREADS_PER_BLOCK>>>(__VA_ARGS__); \
        cudaDeviceSynchronize();

#define KERNEL_CALL_1_1(func, ...)  \
        func<<<1, 1>>>(__VA_ARGS__); \
        cudaDeviceSynchronize();
        
template< typename T >
void checkAndThrowError(T result, char const *const func, const char *const file, int const line)
{
    if (result) {
        DEVICE_RESET
        if (cudaError::cudaErrorInsufficientDriver == result) {
            throw SpecificCudaException(
                "Your graphics driver is not compatible with CUDA 11.2. Please update your Nvidia graphics driver and restart.");
        } else if (cudaError::cudaErrorOperatingSystem == result) {
            throw SpecificCudaException("An operating system call within the CUDA api failed. Please check if your "
                                        "monitor is plugged to the correct graphics card.");
        } else if (cudaError::cudaErrorInitializationError == result) {
            throw SpecificCudaException(
                "CUDA could not be initialized. Please check the minimum hardware requirements. If fulfilled please update your Nvidia graphics driver and restart.");
        } else if (cudaError::cudaErrorUnsupportedPtxVersion == result) {
            throw SpecificCudaException("A CUDA error occurred (cudaErrorUnsupportedPtxVersion). Please update your Nvidia graphics driver and restart.");
        } else {
            std::stringstream stream;
            stream << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << "("
                   << _cudaGetErrorEnum(result) << ") \"" << func << "\"";
            throw BugReportException(stream.str().c_str());
        }
    }
}

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define CHECK_FOR_CUDA_ERROR(val) \
    checkAndThrowError( (val), #val, __FILENAME__, __LINE__ )

#define ABORT() *((int*)nullptr) = 1;
