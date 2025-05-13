#include "Objects.cuh"

#include "Object.cuh"

void Objects::init()
{
    cellPointers.init();
    particlePointers.init();
    heap.init();
}

void Objects::free()
{
    cellPointers.free();
    particlePointers.free();
    heap.free();
}

__device__ void Objects::saveNumEntries()
{
    cellPointers.saveNumEntries();
    particlePointers.saveNumEntries();
}
