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

    //auxiliary data
    int locked;	//0 = unlocked, 1 = locked
    int alive;  //0 = dead, 1 == alive

    __device__ __inline__ bool isSelected()
    {
        return _selected;
    }

    __device__ __inline__ void setSelected(bool value)
    {
        _selected = value;
    }

    __device__ __inline__ bool tryLock() { return 0 == atomicExch(&locked, 1); }

    __device__ __inline__ void releaseLock() { atomicExch(&locked, 0); }

private:
    bool _selected;
};