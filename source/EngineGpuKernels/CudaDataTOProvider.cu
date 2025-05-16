#include "CudaDataTOProvider.cuh"

#include "CudaMemoryManager.cuh"

_CudaDataTOProvider::_CudaDataTOProvider() {}

_CudaDataTOProvider::~_CudaDataTOProvider()
{
    if (_dataTO) {
        destroy();
    }
}

DataTO _CudaDataTOProvider::provideDataTO(ArraySizesForTO const& requiredCapacity)
{
    if (_dataTO) {
        if (fits(_dataTO->capacities, requiredCapacity)) {
            setValueToDevice(_dataTO->numCells, 0ull);
            setValueToDevice(_dataTO->numParticles, 0ull);
            setValueToDevice(_dataTO->heapSize, 0ull);
            return *_dataTO;
        } else {
            destroy();
        }
    }
    try {
        DataTO result;
        result.capacities = requiredCapacity;
        CudaMemoryManager::getInstance().acquireMemory(1, result.numCells);
        CudaMemoryManager::getInstance().acquireMemory(1, result.numParticles);
        CudaMemoryManager::getInstance().acquireMemory(1, result.heapSize);
        CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.cellArray, result.cells);
        CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.particleArray, result.particles);
        CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.heap, result.heap);
        setValueToDevice(result.numCells, 0ull);
        setValueToDevice(result.numParticles, 0ull);
        setValueToDevice(result.heapSize, 0ull);

        _dataTO = result;
        return result;

    } catch (std::bad_alloc const&) {
        throw std::runtime_error("There is not sufficient GPU memory available.");
    }
}

bool _CudaDataTOProvider::fits(ArraySizesForTO const& left, ArraySizesForTO const& right) const
{
    return left.cellArray >= right.cellArray && left.particleArray >= right.particleArray
        && left.heap >= right.heap;
}

void _CudaDataTOProvider::destroy()
{
    CudaMemoryManager::getInstance().freeMemory(_dataTO->cells);
    CudaMemoryManager::getInstance().freeMemory(_dataTO->particles);
    CudaMemoryManager::getInstance().freeMemory(_dataTO->heap);
    CudaMemoryManager::getInstance().freeMemory(_dataTO->numCells);
    CudaMemoryManager::getInstance().freeMemory(_dataTO->numParticles);
    CudaMemoryManager::getInstance().freeMemory(_dataTO->heapSize);
}
