#include "AccessDataTOCache.h"

_AccessDataTOCache::_AccessDataTOCache()
{}

_AccessDataTOCache::~_AccessDataTOCache()
{
    if(_dataTO) {
        _dataTO->destroy();
    }
}

DataTO _AccessDataTOCache::getDataTO(ArraySizes const& arraySizes)
{
    if (_dataTO) {
        auto existingArraySizes = getArraySizes(*_dataTO);
        if (fits(existingArraySizes, arraySizes)) {
            *_dataTO->numCells = 0;
            *_dataTO->numParticles = 0;
            *_dataTO->numAuxiliaryData = 0;
            return *_dataTO;
        } else {
            _dataTO->destroy();
        }
    }
    try {
        DataTO result;
        result.init(arraySizes);
        _dataTO = result;
        return result;
    } catch (std::bad_alloc const&) {
        throw std::runtime_error("There is not sufficient CPU memory available.");
    }
}

bool _AccessDataTOCache::fits(ArraySizes const& left, ArraySizes const& right) const
{
    return left.cellArraySize >= right.cellArraySize && left.particleArraySize >= right.particleArraySize
        && left.rawMemorySize >= right.rawMemorySize;
}

auto _AccessDataTOCache::getArraySizes(DataTO const& dataTO) const -> ArraySizes
{
    return {*dataTO.numCells, *dataTO.numParticles, *dataTO.numAuxiliaryData};
}
