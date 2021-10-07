#pragma once

#include "DllExport.h"

struct GpuConstants
{
    int NUM_THREADS_PER_BLOCK = 64;
    int NUM_BLOCKS = 1024;

    bool operator==(GpuConstants const& other) const
    {
        return NUM_THREADS_PER_BLOCK == other.NUM_THREADS_PER_BLOCK && NUM_BLOCKS == other.NUM_BLOCKS;
    }

    bool operator!=(GpuConstants const& other) const { return !operator==(other); }
};

namespace Const
{
    int const MetadataMemorySize = 50000000; //~ 50 MB should sufficent
}

