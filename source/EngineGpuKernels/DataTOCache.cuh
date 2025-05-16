#pragma once

#include <optional>

#include "EngineInterface/ArraySizesForTO.h"
#include "EngineGpuKernels/ObjectTO.cuh"

class _DataTOCache
{
public:
    _DataTOCache();
    ~_DataTOCache();

    DataTO provideDataTO(ArraySizesForTO const& requiredCapacity);

private:
    bool fits(ArraySizesForTO const& left, ArraySizesForTO const& right) const;
    void destroy();

    std::optional<DataTO> _dataTO;
};

