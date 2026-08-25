#pragma once

#include <algorithm>
#include <chrono>

#include <Base/GlobalSettings.h>
#include <Base/KernelProfiler.h>
#include <Base/KernelTracer.h>

#include <EngineInterface/EngineConstants.h>

#include "Macros.cuh"

// How a kernel is spread over the device. The values are derived from the GPU at startup rather than configured, see
// KernelLaunchSettings.
struct LaunchConfig
{
    int gridSize = 1;
    int blockSize = 1;

    bool operator==(LaunchConfig const& other) const = default;
    auto operator<=>(LaunchConfig const& other) const = default;
};

// Launches a kernel and, in debug mode, times and traces it. Debug mode also synchronizes after every launch, which is
// what makes a crashing or hanging kernel identifiable; the regular path adds nothing to the launch.
//
// Call it through the KERNEL macro, which supplies the name for the trace: launchKernel(KERNEL(cudaFoo), config, data).
template <typename Kernel, typename... Args>
void launchKernel(char const* name, Kernel kernel, LaunchConfig const& config, cudaStream_t stream, Args const&... args)
{
    if (!GlobalSettings::get().isDebugMode()) {
        kernel<<<config.gridSize, config.blockSize, 0, stream>>>(args...);
        return;
    }
    KernelTracer::get().traceBegin(name);
    auto const start = std::chrono::steady_clock::now();
    kernel<<<config.gridSize, config.blockSize, 0, stream>>>(args...);
    if (stream != nullptr) {
        CHECK_FOR_DEVICE_ERRORS(cudaStreamSynchronize(stream));
    } else {
        CHECK_FOR_DEVICE_ERRORS(cudaDeviceSynchronize());
    }
    auto const duration = std::chrono::steady_clock::now() - start;
    KernelProfiler::get().record(name, duration, config.gridSize, config.blockSize);
    KernelTracer::get().traceEnd(duration);
}

// Same on the default stream, for the kernels that run outside the captured timestep graph. It needs a name of its
// own: an overload would be ambiguous with the one above, whose stream is just another argument to deduce.
template <typename Kernel, typename... Args>
void launchKernelOnDefaultStream(char const* name, Kernel kernel, LaunchConfig const& config, Args const&... args)
{
    launchKernel(name, kernel, config, cudaStream_t{nullptr}, args...);
}

#define KERNEL(kernel) #kernel, kernel
