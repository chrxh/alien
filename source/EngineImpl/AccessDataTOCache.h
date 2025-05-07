#pragma once

#include "Base/Definitions.h"

#include "EngineInterface/ArraySizes.h"
#include "EngineGpuKernels/ObjectTO.cuh"

#include "Definitions.h"

class _AccessDataTOCache
{
public:
    _AccessDataTOCache();
    ~_AccessDataTOCache();

    DataTO getDataTO(ArraySizes const& arraySizes);

private:
    bool fits(ArraySizes const& left, ArraySizes const& right) const;
    ArraySizes getArraySizes(DataTO const& dataTO) const;

    std::optional<DataTO> _dataTO;
};

