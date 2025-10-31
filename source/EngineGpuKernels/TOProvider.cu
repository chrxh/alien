#include "TOProvider.cuh"

#include <stdexcept>

_TOProvider::_TOProvider()
{}

_TOProvider::~_TOProvider()
{
    if (_to) {
        destroy(_to.value());
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

TO _TOProvider::provideDataTO(ArraySizesForTO const& requiredCapacity)
{
    try {
        if (_to.has_value()) {
            checkAndExtendCapacity(_to->cells, *_to->numCells, _to->capacities.cells, requiredCapacity.cells);
            checkAndExtendCapacity(_to->particles, *_to->numParticles, _to->capacities.particles, requiredCapacity.particles);
            checkAndExtendCapacity(_to->creatures, *_to->numCreatures, _to->capacities.creatures, requiredCapacity.creatures);
            checkAndExtendCapacity(_to->genomes, *_to->numGenomes, _to->capacities.genomes, requiredCapacity.genomes);
            checkAndExtendCapacity(_to->genes, *_to->numGenes, _to->capacities.genes, requiredCapacity.genes);
            checkAndExtendCapacity(_to->nodes, *_to->numNodes, _to->capacities.nodes, requiredCapacity.nodes);
            checkAndExtendCapacity(_to->heap, *_to->heapSize, _to->capacities.heap, requiredCapacity.heap);
        } else {
            _to = provideNewUnmanagedDataTO(requiredCapacity);
        }
        return _to.value();
    } catch (std::bad_alloc const&) {
        throw std::runtime_error("There is not sufficient CPU memory available.");
    }
}

TO _TOProvider::provideNewUnmanagedDataTO(ArraySizesForTO const& requiredCapacity)
{
    try {
        TO result;

        result.capacities = requiredCapacity;

        result.numCells = new uint64_t;
        result.numParticles = new uint64_t;
        result.numCreatures = new uint64_t;
        result.numGenomes = new uint64_t;
        result.numGenes = new uint64_t;
        result.numNodes = new uint64_t;
        result.heapSize = new uint64_t;

        *result.numCells = 0;
        *result.numParticles = 0;
        *result.numCreatures = 0;
        *result.numGenomes = 0;
        *result.numGenes = 0;
        *result.numNodes = 0;
        *result.heapSize = 0;

        result.cells = new CellTO[requiredCapacity.cells];
        result.particles = new ParticleTO[requiredCapacity.particles];
        result.creatures = new CreatureTO[requiredCapacity.creatures];
        result.genomes = new GenomeTO[requiredCapacity.genomes];
        result.genes = new GeneTO[requiredCapacity.genes];
        result.nodes = new NodeTO[requiredCapacity.nodes];
        result.heap = new uint8_t[requiredCapacity.heap];

        return result;

    } catch (std::bad_alloc const&) {
        throw std::runtime_error("There is not sufficient CPU memory available.");
    }
}

void _TOProvider::destroyUnmanagedDataTO(TO const& dataTO)
{
    destroy(dataTO);
}

void _TOProvider::destroy(TO const& dataTO)
{
    delete dataTO.numCells;
    delete dataTO.numParticles;
    delete dataTO.numCreatures;
    delete dataTO.numGenomes;
    delete dataTO.numGenes;
    delete dataTO.numNodes;
    delete dataTO.heapSize;

    delete[] dataTO.cells;
    delete[] dataTO.particles;
    delete[] dataTO.creatures;
    delete[] dataTO.genomes;
    delete[] dataTO.genes;
    delete[] dataTO.nodes;
    delete[] dataTO.heap;
}
