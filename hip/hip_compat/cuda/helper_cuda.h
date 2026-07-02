#pragma once

// HIP build shim for the project's vendored <cuda/helper_cuda.h>. The upstream
// NVIDIA helper carries an SM-version-to-core table and driver-mode probing
// that does not parse under hipcc; the project only ever calls checkCudaErrors,
// so the HIP build provides just that over hipError_t.

#include <cstdio>
#include <cstdlib>

#include <cuda_to_hip.h>

template <typename T>
inline void check(T result, char const* func, char const* file, int line)
{
    if (result) {
        fprintf(
            stderr,
            "HIP error at %s:%d code=%d \"%s\" \"%s\"\n",
            file,
            line,
            static_cast<int>(result),
            hipGetErrorString(static_cast<hipError_t>(result)),
            func);
        hipDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#ifndef DEVICE_RESET
#define DEVICE_RESET hipDeviceReset();
#endif
