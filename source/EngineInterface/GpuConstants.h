#pragma once

#include "DllExport.h"

struct GpuConstants
{
    int NUM_THREADS_PER_BLOCK = 64;
    int NUM_BLOCKS = 1024;

    int MAX_CELLS = 2000000;
    int MAX_PARTICLES = 2000000;
    int MAX_TOKENS = 500000;

    int MAX_CELLPOINTERS = MAX_CELLS * 10;
    int MAX_PARTICLEPOINTERS = MAX_PARTICLES * 10;
    int MAX_TOKENPOINTERS = MAX_TOKENS * 10;

    int METADATA_DYNAMIC_MEMORY_SIZE = 0;
};
