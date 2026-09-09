#include "KernelLaunchSettingsService.cuh"

#include <algorithm>
#include <ranges>

#include <cuda_runtime.h>

#include <Base/LoggingService.h>

#include <EngineInterface/EngineConstants.h>

#include <EngineKernels/Base.cuh>
#include <EngineKernels/SimulationKernels.cuh>

namespace
{
    // Accounts for the register and shared memory budget of the actual kernel, including on future architectures
    int getBlocksPerMultiProcessor(int threadsPerBlock)
    {
        int result = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&result, cudaNextTimestep_physics_calcFluidForces, threadsPerBlock, 0) != cudaSuccess) {
            return 0;
        }
        return result;
    }
}

KernelLaunchSettings KernelLaunchSettingsService::deriveFromDevice(int deviceNumber) const
{
    KernelLaunchSettings result;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceNumber) != cudaSuccess) {
        return result;
    }
    result.numBlocks = calcNumBlocks(prop.multiProcessorCount);
    result.fluidWarpsPerBlock = calcFluidWarpsPerBlock();

    log(Priority::Important,
        "kernel launch: " + std::to_string(result.numBlocks) + " blocks, " + std::to_string(result.fluidWarpsPerBlock) + " warps per fluid block ("
            + std::to_string(prop.multiProcessorCount) + " multiprocessors, "
            + std::to_string(getBlocksPerMultiProcessor(result.fluidWarpsPerBlock * WARP_SIZE) * result.fluidWarpsPerBlock) + " resident warps each)");
    return result;
}

int KernelLaunchSettingsService::calcNumBlocks(int multiProcessorCount) const
{
    auto constexpr BlocksPerMultiProcessor = 128;
    auto constexpr MinBlocks = 1024;
    return std::max(MinBlocks, multiProcessorCount * BlocksPerMultiProcessor);
}

std::vector<int> KernelLaunchSettingsService::calcFluidResidentWarps() const
{
    std::vector<int> result;
    for (int warpsPerBlock = 1; warpsPerBlock <= MAX_FLUID_WARPS_PER_BLOCK; ++warpsPerBlock) {
        result.emplace_back(getBlocksPerMultiProcessor(warpsPerBlock * WARP_SIZE) * warpsPerBlock);
    }
    return result;
}

// Maximizes resident warps: with one warp per block the blocks-per-multiprocessor limit caps them below the warp
// slots. max_element takes the first maximum, so ties go to the smallest block, which divides the grid more evenly.
int KernelLaunchSettingsService::calcFluidWarpsPerBlock() const
{
    auto const residentWarps = calcFluidResidentWarps();
    return toInt(std::distance(residentWarps.begin(), std::ranges::max_element(residentWarps))) + 1;
}
