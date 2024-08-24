#pragma once

struct GpuSettings
{
    int numBlocks = 16384;

    bool operator==(GpuSettings const& other) const
    {
        return numBlocks == other.numBlocks;
    }

    bool operator!=(GpuSettings const& other) const { return !operator==(other); }
};

