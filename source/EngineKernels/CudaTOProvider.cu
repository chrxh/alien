#include <ranges>

#include "Base.cuh"
#include "CudaMemoryManager.cuh"
#include "CudaTOProvider.cuh"

_CudaTOProvider::_CudaTOProvider() {}

_CudaTOProvider::~_CudaTOProvider() noexcept
{
    for (auto& to : _toByDevice | std::views::values) {
        try {
            destroy(to);
        } catch (...) {
        }
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
    TOs result;
    try {
        auto findResult = _toByDevice.find(getCurrentDevice());
        if (findResult != _toByDevice.end()) {
            auto& to = findResult->second;
            checkAndExtendCapacity(to.objects, *to.numObjects, to.capacities.objects, requiredCapacity.objects);
            checkAndExtendCapacity(to.energyParticles, *to.numEnergyParticles, to.capacities.energyParticles, requiredCapacity.energyParticles);
            checkAndExtendCapacity(to.creatures, *to.numCreatures, to.capacities.creatures, requiredCapacity.creatures);
            checkAndExtendCapacity(to.genomes, *to.numGenomes, to.capacities.genomes, requiredCapacity.genomes);
            checkAndExtendCapacity(to.genes, *to.numGenes, to.capacities.genes, requiredCapacity.genes);
            checkAndExtendCapacity(to.nodes, *to.numNodes, to.capacities.nodes, requiredCapacity.nodes);
            checkAndExtendCapacity(to.heap, *to.heapSize, to.capacities.heap, requiredCapacity.heap);
            result = to;
        } else {
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

            _toByDevice.emplace(getCurrentDevice(), result);
        }
    } catch (...) {
        throw std::runtime_error("GPU memory could not be allocated.");
    }
    return result;
}

void _CudaTOProvider::destroy(TOs& to)
{
    CudaMemoryManager::getInstance().freeMemory(to.objects);
    CudaMemoryManager::getInstance().freeMemory(to.energyParticles);
    CudaMemoryManager::getInstance().freeMemory(to.creatures);
    CudaMemoryManager::getInstance().freeMemory(to.genomes);
    CudaMemoryManager::getInstance().freeMemory(to.genes);
    CudaMemoryManager::getInstance().freeMemory(to.nodes);
    CudaMemoryManager::getInstance().freeMemory(to.heap);

    CudaMemoryManager::getInstance().freeMemory(to.numObjects);
    CudaMemoryManager::getInstance().freeMemory(to.numEnergyParticles);
    CudaMemoryManager::getInstance().freeMemory(to.numCreatures);
    CudaMemoryManager::getInstance().freeMemory(to.numGenomes);
    CudaMemoryManager::getInstance().freeMemory(to.numGenes);
    CudaMemoryManager::getInstance().freeMemory(to.numNodes);
    CudaMemoryManager::getInstance().freeMemory(to.heapSize);
}
