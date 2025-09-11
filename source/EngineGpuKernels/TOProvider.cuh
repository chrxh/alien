#pragma once

#include <optional>

#include "EngineInterface/ArraySizesForTO.h"
#include "EngineGpuKernels/TO.cuh"

class _TOProvider
{
public:
    _TOProvider();
    ~_TOProvider();

    TO provideDataTO(ArraySizesForTO const& requiredCapacity);
    TO provideNewUnmanagedDataTO(ArraySizesForTO const& requiredCapacity);

    static void destroyUnmanagedDataTO(TO const& dataTO);

private:
    static void destroy(TO const& dataTO);

    std::optional<TO> _to;
};

