#pragma once

#include "Base/Definitions.h"

#include "EngineInterface/ObjectTOArraySizes.h"
#include "EngineGpuKernels/ObjectTO.cuh"

#include "Definitions.h"

class _AccessDataTOCache
{
public:
    _AccessDataTOCache();
    ~_AccessDataTOCache();

    DataTO getDataTO(ObjectTOArraySizes const& arraySizes);

private:
    bool fits(ObjectTOArraySizes const& left, ObjectTOArraySizes const& right) const;
    ObjectTOArraySizes getArraySizes(DataTO const& dataTO) const;

    std::optional<DataTO> _dataTO;
};

