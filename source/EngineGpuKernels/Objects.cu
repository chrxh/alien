#include "Objects.cuh"

#include "Object.cuh"

void Objects::init()
{
    cellPointers.init();
    particlePointers.init();
    rawMemory.init();
}

void Objects::free()
{
    cellPointers.free();
    particlePointers.free();
    rawMemory.free();
}

__device__ void Objects::saveNumEntries()
{
    cellPointers.saveNumEntries();
    particlePointers.saveNumEntries();
}
