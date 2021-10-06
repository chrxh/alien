#pragma once

#include "DllExport.h"

struct GpuConstants
{
    int NUM_THREADS_PER_BLOCK = 64;
    int NUM_BLOCKS = 1024;

    int METADATA_DYNAMIC_MEMORY_SIZE = 0;
};
