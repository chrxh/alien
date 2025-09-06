#include "Objects.cuh"

#include "Object.cuh"

void Objects::init()
{
    cells.init();
    particles.init();
    heap.init();
}

void Objects::free()
{
    cells.free();
    particles.free();
    heap.free();
}

__device__ void Objects::saveNumEntries()
{
    cells.saveNumEntries();
    particles.saveNumEntries();
}
