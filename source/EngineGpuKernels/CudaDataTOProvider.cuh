#pragma once

#include <optional>

#include "EngineInterface/ArraySizesForTO.h"

#include "ObjectTO.cuh"

class _CudaDataTOProvider
{
public:
    _CudaDataTOProvider();
    ~_CudaDataTOProvider();

    DataTO provideDataTO(ArraySizesForTO const& requiredCapacity);

private:
    bool fits(ArraySizesForTO const& left, ArraySizesForTO const& right) const;
    void destroy();

    std::optional<DataTO> _dataTO;
};

