#include "Objects.cuh"

#include "Object.cuh"

void Objects::init()
{
    cellPointers.init();
    particles.init();
    particlePointers.init();
    rawMemory.init();
}

void Objects::free()
{
    cellPointers.free();
    particles.free();
    particlePointers.free();
    rawMemory.free();
}

__device__ void Objects::saveNumEntries()
{
    cellPointers.saveNumEntries();
    particlePointers.saveNumEntries();
    particles.saveNumEntries();
}
