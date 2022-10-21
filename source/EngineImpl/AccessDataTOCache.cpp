#include "AccessDataTOCache.h"

_AccessDataTOCache::_AccessDataTOCache(GpuSettings const& gpuConstants)
    : _gpuConstants(gpuConstants)
{}

_AccessDataTOCache::~_AccessDataTOCache()
{
    if(_dataTO) {
        deleteDataTO(*_dataTO);
    }
}

DataTO _AccessDataTOCache::getDataTO(ArraySizes const& arraySizes)
{
    if (_dataTO) {
        auto existingArraySizes = getArraySizes(*_dataTO);
        if (fits(existingArraySizes, arraySizes)) {
            *_dataTO->numCells = 0;
            *_dataTO->numParticles = 0;
            *_dataTO->numAdditionalData = 0;
            return *_dataTO;
        } else {
            deleteDataTO(*_dataTO);
        }
    }
    try {
        DataTO result;
        result.numCells = new uint64_t;
        result.numParticles = new uint64_t;
        result.numAdditionalData = new uint64_t;
        result.cells = new CellTO[arraySizes.cellArraySize];
        result.particles = new ParticleTO[arraySizes.particleArraySize];
        result.additionalData = new uint8_t[arraySizes.additionalDataSize];
        _dataTO = result;
        return result;
    } catch (std::bad_alloc const&) {
        throw BugReportException("There is not sufficient CPU memory available.");
    }
}

bool _AccessDataTOCache::fits(ArraySizes const& left, ArraySizes const& right) const
{
    return left.cellArraySize >= right.cellArraySize && left.particleArraySize >= right.particleArraySize
        && left.additionalDataSize >= right.additionalDataSize;
}

auto _AccessDataTOCache::getArraySizes(DataTO const& dataTO) const -> ArraySizes
{
    return {*dataTO.numCells, *dataTO.numParticles, *dataTO.numAdditionalData};
}

void _AccessDataTOCache::deleteDataTO(DataTO const& dataTO)
{
    delete dataTO.numCells;
    delete dataTO.numParticles;
    delete dataTO.numAdditionalData;
    delete[] dataTO.cells;
    delete[] dataTO.particles;
    delete[] dataTO.additionalData;
}
