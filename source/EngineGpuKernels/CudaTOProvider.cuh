#pragma once

#include <optional>

#include "EngineInterface/ArraySizesForTO.h"

#include "TO.cuh"

class _CudaTOProvider
{
public:
    _CudaTOProvider();
    ~_CudaTOProvider();

    TO provideDataTO(ArraySizesForTO const& requiredCapacity);

private:
    void destroy();

    std::optional<TO> _to;
};

