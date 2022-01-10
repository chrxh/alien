#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/helper_cuda.h>
#include <sstream>

#include "Base/Exceptions.h"

#define KERNEL_CALL(func, ...) func<<<gpuSettings.NUM_BLOCKS, gpuSettings.NUM_THREADS_PER_BLOCK>>>(__VA_ARGS__);
#define KERNEL_CALL_1_1(func, ...) func<<<1, 1>>>(__VA_ARGS__);

//#TODO remove following macros
#define DEPRECATED_KERNEL_CALL(func, ...) func<<<cudaThreadSettings.NUM_BLOCKS, cudaThreadSettings.NUM_THREADS_PER_BLOCK>>>(__VA_ARGS__);
#define DEPRECATED_KERNEL_CALL_1_1(func, ...) func<<<1, 1>>>(__VA_ARGS__);

#define DEPRECATED_KERNEL_CALL_HOST_SYNC(func, ...) \
    func<<<1, 1>>>(__VA_ARGS__); \
    cudaDeviceSynchronize(); \
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());

#define DEPRECATED_KERNEL_CALL_SYNC(func, ...)  \
        func<<<cudaThreadSettings.NUM_BLOCKS, cudaThreadSettings.NUM_THREADS_PER_BLOCK>>>(__VA_ARGS__); \
        cudaDeviceSynchronize();

#define DEPRECATED_KERNEL_CALL_SYNC_1_1(func, ...)  \
        func<<<1, 1>>>(__VA_ARGS__); \
        cudaDeviceSynchronize();

template< typename T >
void checkAndThrowError(T result, char const *const func, const char *const file, int const line)
{
    if (result) {
        DEVICE_RESET
        if (cudaError::cudaErrorInsufficientDriver == result) {
            throw SpecificCudaException(
                "Your graphics driver is not compatible with the required CUDA version. Please update your Nvidia graphics driver and restart.");
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

#define FP_PRECISION 0.00001

#define CUDA_THROW_NOT_IMPLEMENTED() \
    printf("not implemented"); \
    asm("trap;");
