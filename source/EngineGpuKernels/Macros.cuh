#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/helper_cuda.h>
#include <sstream>

#include "Base/Exceptions.h"
#include "Base/GlobalSettings.h"
#include "Base/LoggingService.h"

template< typename T >
void checkAndThrowError(T result, char const *const func, const char *const file, int const line)
{
    if (result) {
        DEVICE_RESET
        std::stringstream stream;
        switch (result) {
        case cudaError::cudaErrorInsufficientDriver:
            stream << "Your graphics driver is not compatible with the required CUDA version. Please update your NVIDIA graphics driver and restart.";
            break;
        case cudaError::cudaErrorOperatingSystem:
            stream << "An operating system call within the CUDA API failed. Please check if your monitor is plugged to the correct graphics card.";
            break;
        case cudaError::cudaErrorInitializationError:
            stream
                << "CUDA could not be initialized. Please check the minimum hardware requirements. If fulfilled please update your NVIDIA graphics driver and "
                   "restart.";
            break;
        case cudaError::cudaErrorUnsupportedPtxVersion:
            stream << "A CUDA error occurred (cudaErrorUnsupportedPtxVersion). Please update your NVIDIA graphics driver and restart.";
            break;
        case cudaError::cudaErrorMemoryAllocation:
            stream << "A CUDA error occurred while allocating memory. A possible reason could be that there is not enough memory available.";
            break;
        default: {
            stream << "CUDA error.";
        }
            break;
        }
        stream << " Location: " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << "(" << _cudaGetErrorEnum(result) << ") \"" << func
               << "\"";
        auto text = stream.str();
        log(Priority::Important, text);

        if (cudaError::cudaErrorMemoryAllocation == result) {
            throw CudaMemoryAllocationException(text);
        } else {
            throw CudaException(text);
        }
    }
}

#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)

#define CHECK_FOR_CUDA_ERROR(val) \
    checkAndThrowError( (val), #val, __FILENAME__, __LINE__ )

#define ABORT() asm("trap;");

#define NEAR_ZERO 0.00001f

#define CHECK(condition) \
    if (!(condition)) { \
        printf("Check failed. File: %s, Line: %d\n", __FILE__, __LINE__); \
        ABORT(); \
    }

#define CUDA_THROW_NOT_IMPLEMENTED() \
    printf("Not implemented error. File: %s, Line: %d\n", __FILE__, __LINE__); \
    ABORT();

#define KERNEL_CALL(func, ...) \
    if (GlobalSettings::getInstance().isDebugMode()) { \
        func<<<gpuSettings.numBlocks, gpuSettings.numThreadsPerBlock>>>(__VA_ARGS__); \
        cudaDeviceSynchronize(); \
        CHECK_FOR_CUDA_ERROR(cudaGetLastError()); \
    } \
    else { \
        func<<<gpuSettings.numBlocks, gpuSettings.numThreadsPerBlock>>>(__VA_ARGS__); \
    }

#define KERNEL_CALL_1_1(func, ...) \
    if (GlobalSettings::getInstance().isDebugMode()) { \
        func<<<1, 1>>>(__VA_ARGS__); \
        cudaDeviceSynchronize(); \
        CHECK_FOR_CUDA_ERROR(cudaGetLastError()); \
    } else { \
        func<<<1, 1>>>(__VA_ARGS__); \
    }
