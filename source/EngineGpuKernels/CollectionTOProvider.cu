#include "CollectionTOProvider.cuh"

#include <stdexcept>

_CollectionTOProvider::_CollectionTOProvider()
{}

_CollectionTOProvider::~_CollectionTOProvider()
{
    if (_collectionTO) {
        destroy(_collectionTO.value());
    }
}

namespace
{
    template <typename T>
    void checkAndExtendCapacity(T*& array, uint64_t& actualSize, uint64_t& actualCapacity, uint64_t requiredCapacity)
    {
        if (actualCapacity < requiredCapacity) {
            delete[] array;
            array = new T[requiredCapacity];
            actualCapacity = requiredCapacity;
            actualSize = 0;
        }
    };
}

CollectionTO _CollectionTOProvider::provideDataTO(ArraySizesForTO const& requiredCapacity)
{
    try {
        if (_collectionTO.has_value()) {
            checkAndExtendCapacity(_collectionTO->cells, *_collectionTO->numCells, _collectionTO->capacities.cells, requiredCapacity.cells);
            checkAndExtendCapacity(_collectionTO->particles, *_collectionTO->numParticles, _collectionTO->capacities.particles, requiredCapacity.particles);
            checkAndExtendCapacity(_collectionTO->genomes, *_collectionTO->numGenomes, _collectionTO->capacities.genomes, requiredCapacity.genomes);
            checkAndExtendCapacity(_collectionTO->genes, *_collectionTO->numGenes, _collectionTO->capacities.genes, requiredCapacity.genes);
            checkAndExtendCapacity(_collectionTO->nodes, *_collectionTO->numNodes, _collectionTO->capacities.nodes, requiredCapacity.nodes);
            checkAndExtendCapacity(_collectionTO->heap, *_collectionTO->heapSize, _collectionTO->capacities.heap, requiredCapacity.heap);
        } else {
            _collectionTO = provideNewUnmanagedDataTO(requiredCapacity);
        }
        return _collectionTO.value();
    } catch (std::bad_alloc const&) {
        throw std::runtime_error("There is not sufficient CPU memory available.");
    }
}

CollectionTO _CollectionTOProvider::provideNewUnmanagedDataTO(ArraySizesForTO const& requiredCapacity)
{
    try {
        CollectionTO result;

        result.capacities = requiredCapacity;

        result.numCells = new uint64_t;
        result.numParticles = new uint64_t;
        result.numGenomes = new uint64_t;
        result.numGenes = new uint64_t;
        result.numNodes = new uint64_t;
        result.heapSize = new uint64_t;

        *result.numCells = 0;
        *result.numParticles = 0;
        *result.numGenomes = 0;
        *result.numGenes = 0;
        *result.numNodes = 0;
        *result.heapSize = 0;

        result.cells = new CellTO[requiredCapacity.cells];
        result.particles = new ParticleTO[requiredCapacity.particles];
        result.genomes = new GenomeTO[requiredCapacity.genomes];
        result.genes = new GeneTO[requiredCapacity.genes];
        result.nodes = new NodeTO[requiredCapacity.nodes];
        result.heap = new uint8_t[requiredCapacity.heap];

        return result;

    } catch (std::bad_alloc const&) {
        throw std::runtime_error("There is not sufficient CPU memory available.");
    }
}

void _CollectionTOProvider::destroyUnmanagedDataTO(CollectionTO const& dataTO)
{
    destroy(dataTO);
}

void _CollectionTOProvider::destroy(CollectionTO const& dataTO)
{
    delete dataTO.numCells;
    delete dataTO.numParticles;
    delete dataTO.numGenomes;
    delete dataTO.numGenes;
    delete dataTO.numNodes;
    delete dataTO.heapSize;

    delete[] dataTO.cells;
    delete[] dataTO.particles;
    delete[] dataTO.genomes;
    delete[] dataTO.genes;
    delete[] dataTO.nodes;
    delete[] dataTO.heap;
}
