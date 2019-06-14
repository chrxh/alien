#pragma once

#include "Base.cuh"

struct Particle
{
    uint64_t id;
    float energy;
    float2 pos;
    float2 vel;

    //auxiliary data
    int locked;	//0 = unlocked, 1 = locked
    bool alive;
};