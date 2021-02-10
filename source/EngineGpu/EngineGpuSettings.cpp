#include "EngineGpuSettings.h"

CudaConstants EngineGpuSettings::getDefaultCudaConstants()
{
    CudaConstants result;
    result.NUM_THREADS_PER_BLOCK = 32;
    result.NUM_BLOCKS = 512;
    result.MAX_CLUSTERS = 100000;
    result.MAX_CLUSTERPOINTERS = result.MAX_CLUSTERS * 10;
    result.MAX_CELLS = 500000;
    result.MAX_CELLPOINTERS = result.MAX_CELLS * 10;
    result.MAX_TOKENS = 10000;
    result.MAX_TOKENPOINTERS = result.MAX_TOKENS * 10;
    result.MAX_PARTICLES = 1000000;
    result.MAX_PARTICLEPOINTERS = result.MAX_PARTICLES;
    result.DYNAMIC_MEMORY_SIZE = 50000000;
    result.METADATA_DYNAMIC_MEMORY_SIZE = 10000000;

    return result;
}
