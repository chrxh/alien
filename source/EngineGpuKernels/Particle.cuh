#pragma once

#include "Base.cuh"

struct ParticleMetadata
{
    unsigned char color;
};

struct Particle
{
    uint64_t id;
    float2 absPos;
    float2 vel;
    ParticleMetadata metadata;
    float energy;

    //editing data
    int selected;  //0 = no, 1 = selected

    //auxiliary data
    int locked;	//0 = unlocked, 1 = locked

    __device__ __inline__ bool tryLock() {
        auto result = 0 == atomicExch(&locked, 1);
        if (result) {
            __threadfence();
        }
        return result;
    }

    __device__ __inline__ void releaseLock() {
        __threadfence();
        atomicExch(&locked, 0);
    }
};