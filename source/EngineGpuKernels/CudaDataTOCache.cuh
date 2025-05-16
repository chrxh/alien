#pragma once

#include <optional>

#include "EngineInterface/ArraySizesForTO.h"

#include "ObjectTO.cuh"

class _CudaDataTOCache
{
public:
    _CudaDataTOCache();
    ~_CudaDataTOCache();

    DataTO provideDataTO(ArraySizesForTO const& requiredCapacity);

private:
    bool fits(ArraySizesForTO const& left, ArraySizesForTO const& right) const;
    void destroy();

    std::optional<DataTO> _dataTO;
};

