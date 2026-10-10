#include "Entities.cuh"

void Entities::init()
{
    objects.init();
    energies.init();
    heap.init();
}

void Entities::free()
{
    objects.free();
    energies.free();
    heap.free();
}

__device__ void Entities::saveNumEntries()
{
    objects.saveNumEntries();
    energies.saveNumEntries();
}
