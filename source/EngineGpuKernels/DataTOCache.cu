#include "DataTOCache.cuh"

#include <stdexcept>

_DataTOCache::_DataTOCache()
{}

_DataTOCache::~_DataTOCache()
{
    if (_dataTO) {
        destroy(_dataTO.value());
    }
}

DataTO _DataTOCache::provideDataTO(ArraySizesForTO const& requiredCapacity)
{
    if (_dataTO) {
        if (fits(_dataTO->capacities, requiredCapacity)) {
            *_dataTO->numCells = 0;
            *_dataTO->numParticles = 0;
            *_dataTO->heapSize = 0;
            return _dataTO.value();
        } else {
            destroy(_dataTO.value());
        }
    }
    try {
        DataTO result;
        result.capacities = requiredCapacity;
        result.numCells = new uint64_t;
        result.numParticles = new uint64_t;
        result.heapSize = new uint64_t;
        *result.numCells = 0;
        *result.numParticles = 0;
        *result.heapSize = 0;
        result.cells = new CellTO[requiredCapacity.cellArray];
        result.particles = new ParticleTO[requiredCapacity.particleArray];
        result.heap = new uint8_t[requiredCapacity.heap];
        _dataTO = result;
        return result;
    } catch (std::bad_alloc const&) {
        throw std::runtime_error("There is not sufficient CPU memory available.");
    }
}

DataTO _DataTOCache::provideNewUnmanagedDataTO(ArraySizesForTO const& requiredCapacity)
{
    try {
        DataTO result;
        result.capacities = requiredCapacity;
        result.numCells = new uint64_t;
        result.numParticles = new uint64_t;
        result.heapSize = new uint64_t;
        *result.numCells = 0;
        *result.numParticles = 0;
        *result.heapSize = 0;
        result.cells = new CellTO[requiredCapacity.cellArray];
        result.particles = new ParticleTO[requiredCapacity.particleArray];
        result.heap = new uint8_t[requiredCapacity.heap];
        return result;
    } catch (std::bad_alloc const&) {
        throw std::runtime_error("There is not sufficient CPU memory available.");
    }
}

void _DataTOCache::destroyUnmanagedDataTO(DataTO const& dataTO)
{
    destroy(dataTO);
}

bool _DataTOCache::fits(ArraySizesForTO const& left, ArraySizesForTO const& right) const
{
    return left.cellArray >= right.cellArray && left.particleArray >= right.particleArray
        && left.heap >= right.heap;
}

void _DataTOCache::destroy(DataTO const& dataTO)
{
    delete dataTO.numCells;
    delete dataTO.numParticles;
    delete dataTO.heapSize;
    delete[] dataTO.cells;
    delete[] dataTO.particles;
    delete[] dataTO.heap;
}
