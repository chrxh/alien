#include "Entity.cuh"
#include "Entities.cuh"

void Entities::init()
{
    objects.init();
    energyParticles.init();
    heap.init();
}

void Entities::free()
{
    objects.free();
    energyParticles.free();
    heap.free();
}

__device__ void Entities::saveNumEntries()
{
    objects.saveNumEntries();
    energyParticles.saveNumEntries();
}
