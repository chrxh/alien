#pragma once

#include "DllExport.h"

struct GpuConstants
{
    int NUM_THREADS_PER_BLOCK = 64;
    int NUM_BLOCKS = 1024;
};

namespace Const
{
    int const MetadataMemorySize = 50000000; //~ 50 MB should sufficent
}

