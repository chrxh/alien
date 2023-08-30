#pragma once

struct GpuSettings
{
    int numThreadsPerBlock = 8;
    int numBlocks = 16384;

    bool operator==(GpuSettings const& other) const
    {
        return numThreadsPerBlock == other.numThreadsPerBlock && numBlocks == other.numBlocks;
    }

    bool operator!=(GpuSettings const& other) const { return !operator==(other); }
};

