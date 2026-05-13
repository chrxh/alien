#pragma once

#include <optional>

#include <EngineInterface/ArraySizesForTOs.h>

#include "TOs.cuh"

class _CudaTOProvider
{
public:
    _CudaTOProvider();
    ~_CudaTOProvider() noexcept;

    TOs provideDataTO(ArraySizesForTOs const& requiredCapacity);

private:
    void destroy();

    std::optional<TOs> _to;
};
