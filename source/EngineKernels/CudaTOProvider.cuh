#pragma once

#include <map>

#include <EngineInterface/ArraySizesForTOs.h>

#include "TOs.cuh"

// Provides the transfer objects on the current device
class _CudaTOProvider
{
public:
    _CudaTOProvider();
    ~_CudaTOProvider() noexcept;

    TOs provideDataTO(ArraySizesForTOs const& requiredCapacity);

private:
    void destroy(TOs& to);

    std::map<int, TOs> _toByDevice;
};
