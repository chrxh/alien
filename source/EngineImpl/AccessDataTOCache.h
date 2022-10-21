#pragma once

#include "Base/Definitions.h"

#include "EngineInterface/ArraySizes.h"
#include "EngineInterface/GpuSettings.h"
#include "EngineGpuKernels/TOs.cuh"

#include "Definitions.h"

class _AccessDataTOCache
{
public:
    _AccessDataTOCache(GpuSettings const& gpuConstants);
    ~_AccessDataTOCache();

    DataTO getDataTO(ArraySizes const& arraySizes);

private:
    bool fits(ArraySizes const& left, ArraySizes const& right) const;
    ArraySizes getArraySizes(DataTO const& dataTO) const;
    void deleteDataTO(DataTO const& dataTO);

    GpuSettings _gpuConstants;
    std::optional<DataTO> _dataTO;
};

