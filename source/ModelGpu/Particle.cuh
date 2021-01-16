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

    __device__ __inline__ float getEnergy_safe()
    {
        return atomicAdd(&_energy, 0);
    }

    __device__ __inline__ float getEnergy() const
    {
        return _energy;
    }

    __device__ __inline__ void setEnergy_safe(float value)
    {
        atomicExch(&_energy, value);
    }

    __device__ __inline__ void setEnergy(float value)
    {
        _energy = value;
    }

    __device__ __inline__ void changeEnergy(float changeValue)
    {
        atomicAdd(&_energy, changeValue);
    }

    __device__ __inline__ bool isSelected()
    {
        return _selected;
    }

    __device__ __inline__ void setSelected(bool value)
    {
        _selected = value;
    }

private:
    float _energy;
    bool _selected;
};