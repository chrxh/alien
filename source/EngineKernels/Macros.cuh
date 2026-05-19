#pragma once

#include <cassert>
#include <sstream>
#include <string>
#include <vector>

#include <Base/AlienExceptions.h>
#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>
#include <Base/Singleton.h>

#include <cuda/helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class CudaContextState
{
    MAKE_SINGLETON(CudaContextState);

public:
    void setInvalid() { _invalid = true; }
    void reset() { _invalid = false; }
    bool isInvalid() const { return _invalid; }

private:
    bool _invalid = false;
};

template <typename T>
void checkAndThrowError(T result)
{
    if (result) {
        CudaContextState::get().setInvalid();
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
        case cudaError::cudaErrorIllegalAddress:
            stream << "A CUDA error occurred (cudaErrorIllegalAddress).";
            break;
        case cudaError::cudaErrorLaunchFailure:
            stream << "A CUDA error occurred (cudaErrorLaunchFailure).";
            break;
        default: {
            stream << "CUDA error.";
        } break;
        }
        stream << " Error code: " << result;
        auto text = stream.str();
        log(Priority::Important, text);

        if (cudaError::cudaErrorMemoryAllocation == result) {
            throw CudaMemoryAllocationException(text);
        } else {
            throw AlienException(text);
        }
    }
}

#define CHECK_FOR_DEVICE_ERRORS(val) checkAndThrowError((val))

#define ABORT() (*((int*)0) = 0)

#define NEAR_ZERO 1.0e-4f

#define DEVICE_CHECK(condition) \
    if (!(condition)) { \
        printf("Check failed. File: %s, Line: %d\n", __FILE__, __LINE__); \
        ABORT(); \
    }

#define DEVICE_THROW_NOT_IMPLEMENTED() \
    printf("Not implemented error. File: %s, Line: %d\n", __FILE__, __LINE__); \
    ABORT();

#define KERNEL_CALL(func, ...) \
    if (GlobalSettings::get().isDebugMode()) { \
        func<<<gpuSettings.numBlocks, 8>>>(__VA_ARGS__); \
        CHECK_FOR_DEVICE_ERRORS(cudaDeviceSynchronize()); \
    } else { \
        func<<<gpuSettings.numBlocks, 8>>>(__VA_ARGS__); \
    }

#define KERNEL_CALL_1_1(func, ...) \
    if (GlobalSettings::get().isDebugMode()) { \
        func<<<1, 1>>>(__VA_ARGS__); \
        CHECK_FOR_DEVICE_ERRORS(cudaDeviceSynchronize()); \
    } else { \
        func<<<1, 1>>>(__VA_ARGS__); \
    }

#define KERNEL_CALL_MOD(func, threadsPerBlock, ...) \
    if (GlobalSettings::get().isDebugMode()) { \
        func<<<gpuSettings.numBlocks, threadsPerBlock>>>(__VA_ARGS__); \
        CHECK_FOR_DEVICE_ERRORS(cudaDeviceSynchronize()); \
    } else { \
        func<<<gpuSettings.numBlocks, threadsPerBlock>>>(__VA_ARGS__); \
    }

// Stream-based kernel launch macros for CUDA Graph capture
// In debug mode, synchronize after each kernel for precise crash information
#define STREAM_KERNEL_CALL(func, stream, numBlocks, ...) \
    func<<<numBlocks, 8, 0, stream>>>(__VA_ARGS__); \
    if (GlobalSettings::get().isDebugMode()) { \
        CHECK_FOR_DEVICE_ERRORS(cudaStreamSynchronize(stream)); \
    }

#define STREAM_KERNEL_CALL_1_1(func, stream, ...) \
    func<<<1, 1, 0, stream>>>(__VA_ARGS__); \
    if (GlobalSettings::get().isDebugMode()) { \
        CHECK_FOR_DEVICE_ERRORS(cudaStreamSynchronize(stream)); \
    }

#define STREAM_KERNEL_CALL_MOD(func, stream, numBlocks, threadsPerBlock, ...) \
    func<<<numBlocks, threadsPerBlock, 0, stream>>>(__VA_ARGS__); \
    if (GlobalSettings::get().isDebugMode()) { \
        CHECK_FOR_DEVICE_ERRORS(cudaStreamSynchronize(stream)); \
    }
