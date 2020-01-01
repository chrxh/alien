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

    //auxiliary data
    int locked;	//0 = unlocked, 1 = locked
    int alive;  //0 = dead, 1 == alive

    __device__ __inline__ float getEnergy()
    {
        return atomicAdd(&_energy, 0);
    }

    __device__ __inline__ void setEnergy(float value)
    {
        atomicExch(&_energy, value);
    }

    __device__ __inline__ void changeEnergy(float changeValue)
    {
        atomicAdd(&_energy, changeValue);
    }


private:
    float _energy;
};