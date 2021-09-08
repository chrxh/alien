#include "AccessDataTOCache.h"

_AccessDataTOCache::_AccessDataTOCache(GpuConstants const& gpuConstants)
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

DataAccessTO _AccessDataTOCache::getDataTO()
{
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
        result.cells = new CellAccessTO[_gpuConstants.MAX_CELLS];
        result.particles = new ParticleAccessTO[_gpuConstants.MAX_PARTICLES];
        result.tokens = new TokenAccessTO[_gpuConstants.MAX_TOKENS];
        result.stringBytes = new char[_gpuConstants.METADATA_DYNAMIC_MEMORY_SIZE];
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
