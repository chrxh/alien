#pragma once

#include <algorithm>

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
        auto constexpr WarpsPerBlockWhenBlocksAreScarce = 16;
        auto constexpr EnoughResidentBlocks = 8;
        return blocksPerMultiProcessor >= EnoughResidentBlocks ? 1 : WarpsPerBlockWhenBlocksAreScarce;
    }
};
