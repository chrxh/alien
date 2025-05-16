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
    DataTO provideNewUnmanagedDataTO(ArraySizesForTO const& requiredCapacity);

    static void destroyUnmanagedDataTO(DataTO const& dataTO);

private:
    bool fits(ArraySizesForTO const& left, ArraySizesForTO const& right) const;
    static void destroy(DataTO const& dataTO);

    std::optional<DataTO> _dataTO;
};

