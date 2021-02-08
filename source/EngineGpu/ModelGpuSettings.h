#pragma once

#include "EngineGpuKernels/CudaConstants.h"

class ModelGpuSettings
{
public:
    static CudaConstants getDefaultCudaConstants();
};

