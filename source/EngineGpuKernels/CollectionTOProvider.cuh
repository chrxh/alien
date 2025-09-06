#pragma once

#include <optional>

#include "EngineInterface/ArraySizesForTO.h"
#include "EngineGpuKernels/ObjectTO.cuh"

class _CollectionTOProvider
{
public:
    _CollectionTOProvider();
    ~_CollectionTOProvider();

    CollectionTO provideDataTO(ArraySizesForTO const& requiredCapacity);
    CollectionTO provideNewUnmanagedDataTO(ArraySizesForTO const& requiredCapacity);

    static void destroyUnmanagedDataTO(CollectionTO const& dataTO);

private:
    static void destroy(CollectionTO const& dataTO);

    std::optional<CollectionTO> _collectionTO;
};

