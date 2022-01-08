#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "DynamicMemory.cuh"

struct Entities
{
    Array<Cell*> cellPointers;
    Array<Token*> tokenPointers;
    Array<Particle*> particlePointers;

    Array<Cell> cells;
    Array<Token> tokens;
    Array<Particle> particles;

    DynamicMemory dynamicMemory;

    void init();
    void free();
};

