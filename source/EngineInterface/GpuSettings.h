#pragma once

#include "DllExport.h"

struct GpuSettings
{
    int NUM_THREADS_PER_BLOCK = 64;
    int NUM_BLOCKS = 1024;

    bool operator==(GpuSettings const& other) const
    {
        return NUM_THREADS_PER_BLOCK == other.NUM_THREADS_PER_BLOCK && NUM_BLOCKS == other.NUM_BLOCKS;
    }

    bool operator!=(GpuSettings const& other) const { return !operator==(other); }
};

namespace Const
{
    int const MetadataMemorySize = 50000000; //~ 50 MB should sufficent
}

