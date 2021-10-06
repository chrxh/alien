#pragma once

#include "Base/Definitions.h"

#include "EngineGpuKernels/AccessTOs.cuh"

#include "Definitions.h"

class _AccessDataTOCache
{
public:
    _AccessDataTOCache(GpuConstants const& gpuConstants);
    ~_AccessDataTOCache();

    struct ArraySizes
    {
        int cellArraySize;
        int particleArraySize;
        int tokenArraySize;

        bool operator==(ArraySizes const& other) const
        {
            return cellArraySize == other.cellArraySize && particleArraySize == other.particleArraySize
                && tokenArraySize == other.tokenArraySize;
        }

        bool operator!=(ArraySizes const& other) const { return !operator==(other); };
    };
    DataAccessTO getDataTO(ArraySizes const& arraySizes);
    void releaseDataTO(DataAccessTO const& dataTO);

private:
    DataAccessTO getNewDataTO();
    void deleteDataTO(DataAccessTO const& dataTO);

    GpuConstants _gpuConstants;
    std::vector<DataAccessTO> _freeDataTOs;
    std::vector<DataAccessTO> _usedDataTOs;
    boost::optional<ArraySizes> _arraySizes;
};

