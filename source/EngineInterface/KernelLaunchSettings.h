#pragma once

#include <algorithm>

#include "EngineConstants.h"

struct KernelLaunchSettings
{
    int numBlocks = 16384;

    int fluidWarpsPerBlock = 1;

    bool operator==(KernelLaunchSettings const& other) const = default;

    static int calcNumBlocks(int multiProcessorCount)
    {
        auto constexpr BlocksPerMultiProcessor = 128;
        return std::clamp(multiProcessorCount * BlocksPerMultiProcessor, 1024, 65536);
    }

    static int calcWarpsPerBlock(int blocksPerMultiProcessor)
    {
        auto constexpr EnoughResidentBlocks = 8;
        return blocksPerMultiProcessor >= EnoughResidentBlocks ? 1 : MAX_FLUID_WARPS_PER_BLOCK;
    }
};
