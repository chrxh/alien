#pragma once

#include "Base/Definitions.h"

#include "EngineInterface/ArraySizesForObjectTOs.h"
#include "EngineGpuKernels/ObjectTO.cuh"

#include "Definitions.h"

class _AccessDataTOCache
{
public:
    _AccessDataTOCache();
    ~_AccessDataTOCache();

    DataTO getDataTO(ArraySizesForObjectTOs const& arraySizes);

private:
    bool fits(ArraySizesForObjectTOs const& left, ArraySizesForObjectTOs const& right) const;
    ArraySizesForObjectTOs getArraySizes(DataTO const& dataTO) const;

    std::optional<DataTO> _dataTO;
};

