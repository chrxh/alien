#pragma once

#include <optional>

#include "EngineInterface/ArraySizesForTO.h"

#include "ObjectTO.cuh"

class _CudaCollectionTOProvider
{
public:
    _CudaCollectionTOProvider();
    ~_CudaCollectionTOProvider();

    CollectionTO provideDataTO(ArraySizesForTO const& requiredCapacity);

private:
    void destroy();

    std::optional<CollectionTO> _collectionTO;
};

