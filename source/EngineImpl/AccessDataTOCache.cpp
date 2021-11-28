#include "AccessDataTOCache.h"

_AccessDataTOCache::_AccessDataTOCache(GpuSettings const& gpuConstants)
    : _gpuConstants(gpuConstants)
{}

_AccessDataTOCache::~_AccessDataTOCache()
{
    for (DataAccessTO const& dataTO : _freeDataTOs) {
        deleteDataTO(dataTO);
    }
    for (DataAccessTO const& dataTO : _usedDataTOs) {
        deleteDataTO(dataTO);
    }
}

DataAccessTO _AccessDataTOCache::getDataTO(ArraySizes const& arraySizes)
{
    if (!_arraySizes || * _arraySizes != arraySizes) {
        for (DataAccessTO const& dataTO : _freeDataTOs) {
            deleteDataTO(dataTO);
        }
        for (DataAccessTO const& dataTO : _usedDataTOs) {
            deleteDataTO(dataTO);
        }
        _freeDataTOs.clear();
        _usedDataTOs.clear();
        _arraySizes = arraySizes;
    }

    DataAccessTO result;
    if (!_freeDataTOs.empty()) {
        result = *_freeDataTOs.begin();
        _freeDataTOs.erase(_freeDataTOs.begin());
        _usedDataTOs.emplace_back(result);
        return result;
    }
    result = getNewDataTO();
    _usedDataTOs.emplace_back(result);
    return result;
}

void _AccessDataTOCache::releaseDataTO(DataAccessTO const& dataTO)
{
    auto usedDataTO = std::find_if(_usedDataTOs.begin(), _usedDataTOs.end(), [&dataTO](DataAccessTO const& usedDataTO) {
        return usedDataTO == dataTO;
    });
    if (usedDataTO != _usedDataTOs.end()) {
        _freeDataTOs.emplace_back(*usedDataTO);
        _usedDataTOs.erase(usedDataTO);
    }
}

DataAccessTO _AccessDataTOCache::getNewDataTO()
{
    try {
        DataAccessTO result;
        result.numCells = new int;
        result.numParticles = new int;
        result.numTokens = new int;
        result.numStringBytes = new int;
        result.cells = new CellAccessTO[_arraySizes->cellArraySize];
        result.particles = new ParticleAccessTO[_arraySizes->particleArraySize];
        result.tokens = new TokenAccessTO[_arraySizes->tokenArraySize];
        result.stringBytes = new char[Const::MetadataMemorySize];
        *result.numCells = 0;
        *result.numParticles = 0;
        *result.numTokens = 0;
        *result.numStringBytes = 0;
        return result;
    } catch (std::bad_alloc const&) {
        throw BugReportException("There is not sufficient CPU memory available.");
    }
}

void _AccessDataTOCache::deleteDataTO(DataAccessTO const& dataTO)
{
    delete dataTO.numCells;
    delete dataTO.numParticles;
    delete dataTO.numTokens;
    delete dataTO.numStringBytes;
    delete[] dataTO.cells;
    delete[] dataTO.particles;
    delete[] dataTO.tokens;
    delete[] dataTO.stringBytes;
}
