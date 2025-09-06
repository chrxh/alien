#include "CudaCollectionTOProvider.cuh"

#include "CudaMemoryManager.cuh"

_CudaCollectionTOProvider::_CudaCollectionTOProvider() {}

_CudaCollectionTOProvider::~_CudaCollectionTOProvider()
{
    if (_collectionTO) {
        destroy();
    }
}

namespace
{
    template <typename T>
    void checkAndExtendCapacity(T*& array, uint64_t& actualSize, uint64_t& actualCapacity, uint64_t requiredCapacity)
    {
        if (actualCapacity < requiredCapacity) {
            CudaMemoryManager::getInstance().freeMemory(array);
            CudaMemoryManager::getInstance().acquireMemory(requiredCapacity, array);
            actualCapacity = requiredCapacity;
            setValueToDevice(&actualSize, static_cast<uint64_t>(0));
        }
    };
}

CollectionTO _CudaCollectionTOProvider::provideDataTO(ArraySizesForTO const& requiredCapacity)
{
    try {
        if (_collectionTO.has_value()) {
            checkAndExtendCapacity(_collectionTO->cells, *_collectionTO->numCells, _collectionTO->capacities.cells, requiredCapacity.cells);
            checkAndExtendCapacity(_collectionTO->particles, *_collectionTO->numParticles, _collectionTO->capacities.particles, requiredCapacity.particles);
            checkAndExtendCapacity(_collectionTO->creatures, *_collectionTO->numCreatures, _collectionTO->capacities.creatures, requiredCapacity.creatures);
            checkAndExtendCapacity(_collectionTO->genes, *_collectionTO->numGenes, _collectionTO->capacities.genes, requiredCapacity.genes);
            checkAndExtendCapacity(_collectionTO->nodes, *_collectionTO->numNodes, _collectionTO->capacities.nodes, requiredCapacity.nodes);
            checkAndExtendCapacity(_collectionTO->heap, *_collectionTO->heapSize, _collectionTO->capacities.heap, requiredCapacity.heap);
        } else {
            CollectionTO result;
            result.capacities = requiredCapacity;
            CudaMemoryManager::getInstance().acquireMemory(1, result.numCells);
            CudaMemoryManager::getInstance().acquireMemory(1, result.numParticles);
            CudaMemoryManager::getInstance().acquireMemory(1, result.numCreatures);
            CudaMemoryManager::getInstance().acquireMemory(1, result.numGenes);
            CudaMemoryManager::getInstance().acquireMemory(1, result.numNodes);
            CudaMemoryManager::getInstance().acquireMemory(1, result.heapSize);
            CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.cells, result.cells);
            CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.particles, result.particles);
            CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.creatures, result.creatures);
            CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.genes, result.genes);
            CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.nodes, result.nodes);
            CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.heap, result.heap);
            setValueToDevice(result.numCells, static_cast<uint64_t>(0));
            setValueToDevice(result.numParticles, static_cast<uint64_t>(0));
            setValueToDevice(result.numCreatures, static_cast<uint64_t>(0));
            setValueToDevice(result.numGenes, static_cast<uint64_t>(0));
            setValueToDevice(result.numNodes, static_cast<uint64_t>(0));
            setValueToDevice(result.heapSize, static_cast<uint64_t>(0));

            _collectionTO = result;
        }
        return _collectionTO.value();
    } catch (...) {
        throw std::runtime_error("GPU memory could not be allocated.");
    }
}

void _CudaCollectionTOProvider::destroy()
{
    CudaMemoryManager::getInstance().freeMemory(_collectionTO->cells);
    CudaMemoryManager::getInstance().freeMemory(_collectionTO->particles);
    CudaMemoryManager::getInstance().freeMemory(_collectionTO->creatures);
    CudaMemoryManager::getInstance().freeMemory(_collectionTO->genes);
    CudaMemoryManager::getInstance().freeMemory(_collectionTO->nodes);
    CudaMemoryManager::getInstance().freeMemory(_collectionTO->heap);

    CudaMemoryManager::getInstance().freeMemory(_collectionTO->numCells);
    CudaMemoryManager::getInstance().freeMemory(_collectionTO->numParticles);
    CudaMemoryManager::getInstance().freeMemory(_collectionTO->numCreatures);
    CudaMemoryManager::getInstance().freeMemory(_collectionTO->numGenes);
    CudaMemoryManager::getInstance().freeMemory(_collectionTO->numNodes);
    CudaMemoryManager::getInstance().freeMemory(_collectionTO->heapSize);
}
