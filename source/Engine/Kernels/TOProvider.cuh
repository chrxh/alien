#pragma once

#include <optional>

#include <Engine/Interface/ArraySizesForTOs.h>

#include <Engine/Kernels/TOs.cuh>

class _TOProvider
{
public:
    _TOProvider();
    ~_TOProvider();

    TOs provideDataTO(ArraySizesForTOs const& requiredCapacity);
    TOs provideNewUnmanagedDataTO(ArraySizesForTOs const& requiredCapacity);

    static void destroyUnmanagedDataTO(TOs const& to);

private:
    static void destroy(TOs const& to);

    std::optional<TOs> _to;
};
