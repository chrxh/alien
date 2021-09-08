#pragma once

#include "Base/Definitions.h"

#include "EngineGpuKernels/AccessTOs.cuh"

#include "Definitions.h"

class _AccessDataTOCache
{
public:
    _AccessDataTOCache(GpuConstants const& gpuConstants);
    ~_AccessDataTOCache();

    DataAccessTO getDataTO();
    void releaseDataTO(DataAccessTO const& dataTO);

private:
    DataAccessTO getNewDataTO();
    void deleteDataTO(DataAccessTO const& dataTO);

    GpuConstants _gpuConstants;
    std::vector<DataAccessTO> _freeDataTOs;
    std::vector<DataAccessTO> _usedDataTOs;
};

