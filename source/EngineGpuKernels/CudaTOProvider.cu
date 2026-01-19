#include "CudaMemoryManager.cuh"
#include "CudaTOProvider.cuh"

_CudaTOProvider::_CudaTOProvider() {}

_CudaTOProvider::~_CudaTOProvider()
{
    if (_to) {
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

TOs _CudaTOProvider::provideDataTO(ArraySizesForTOs const& requiredCapacity)
{
    try {
        if (_to.has_value()) {
            checkAndExtendCapacity(_to->objects, *_to->numObjects, _to->capacities.objects, requiredCapacity.objects);
            checkAndExtendCapacity(_to->energyParticles, *_to->numEnergyParticles, _to->capacities.energyParticles, requiredCapacity.energyParticles);
            checkAndExtendCapacity(_to->creatures, *_to->numCreatures, _to->capacities.creatures, requiredCapacity.creatures);
            checkAndExtendCapacity(_to->genomes, *_to->numGenomes, _to->capacities.genomes, requiredCapacity.genomes);
            checkAndExtendCapacity(_to->genes, *_to->numGenes, _to->capacities.genes, requiredCapacity.genes);
            checkAndExtendCapacity(_to->nodes, *_to->numNodes, _to->capacities.nodes, requiredCapacity.nodes);
            checkAndExtendCapacity(_to->heap, *_to->heapSize, _to->capacities.heap, requiredCapacity.heap);
        } else {
            TOs result;
            result.capacities = requiredCapacity;
            CudaMemoryManager::getInstance().acquireMemory(1, result.numObjects);
            CudaMemoryManager::getInstance().acquireMemory(1, result.numEnergyParticles);
            CudaMemoryManager::getInstance().acquireMemory(1, result.numCreatures);
            CudaMemoryManager::getInstance().acquireMemory(1, result.numGenomes);
            CudaMemoryManager::getInstance().acquireMemory(1, result.numGenes);
            CudaMemoryManager::getInstance().acquireMemory(1, result.numNodes);
            CudaMemoryManager::getInstance().acquireMemory(1, result.heapSize);
            CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.objects, result.objects);
            CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.energyParticles, result.energyParticles);
            CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.creatures, result.creatures);
            CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.genomes, result.genomes);
            CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.genes, result.genes);
            CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.nodes, result.nodes);
            CudaMemoryManager::getInstance().acquireMemory(requiredCapacity.heap, result.heap);
            setValueToDevice(result.numObjects, static_cast<uint64_t>(0));
            setValueToDevice(result.numEnergyParticles, static_cast<uint64_t>(0));
            setValueToDevice(result.numCreatures, static_cast<uint64_t>(0));
            setValueToDevice(result.numGenomes, static_cast<uint64_t>(0));
            setValueToDevice(result.numGenes, static_cast<uint64_t>(0));
            setValueToDevice(result.numNodes, static_cast<uint64_t>(0));
            setValueToDevice(result.heapSize, static_cast<uint64_t>(0));

            _to = result;
        }
        return _to.value();
    } catch (...) {
        throw std::runtime_error("GPU memory could not be allocated.");
    }
}

void _CudaTOProvider::destroy()
{
    CudaMemoryManager::getInstance().freeMemory(_to->objects);
    CudaMemoryManager::getInstance().freeMemory(_to->energyParticles);
    CudaMemoryManager::getInstance().freeMemory(_to->creatures);
    CudaMemoryManager::getInstance().freeMemory(_to->genomes);
    CudaMemoryManager::getInstance().freeMemory(_to->genes);
    CudaMemoryManager::getInstance().freeMemory(_to->nodes);
    CudaMemoryManager::getInstance().freeMemory(_to->heap);

    CudaMemoryManager::getInstance().freeMemory(_to->numObjects);
    CudaMemoryManager::getInstance().freeMemory(_to->numEnergyParticles);
    CudaMemoryManager::getInstance().freeMemory(_to->numCreatures);
    CudaMemoryManager::getInstance().freeMemory(_to->numGenomes);
    CudaMemoryManager::getInstance().freeMemory(_to->numGenes);
    CudaMemoryManager::getInstance().freeMemory(_to->numNodes);
    CudaMemoryManager::getInstance().freeMemory(_to->heapSize);
}
