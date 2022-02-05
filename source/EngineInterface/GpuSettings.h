#pragma once

struct GpuSettings
{
    int numThreadsPerBlock = 32;
    int numBlocks = 2048;

    bool operator==(GpuSettings const& other) const
    {
        return numThreadsPerBlock == other.numThreadsPerBlock && numBlocks == other.numBlocks;
    }

    bool operator!=(GpuSettings const& other) const { return !operator==(other); }
};

