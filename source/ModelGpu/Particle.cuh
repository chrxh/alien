#pragma once

#include "Base.cuh"

struct Particle
{
    uint64_t id;
    float2 absPos;
    float2 vel;

    //auxiliary data
    int locked;	//0 = unlocked, 1 = locked
    int alive;  //0 = dead, 1 == alive

    __device__ __inline__ float getEnergy()
    {
        auto const value = atomicAdd(&_energy, 0);
        if (value < 0) {
            printf("Particle::getEnergy negative, value: %f, parameter: \n", value);
            while (true) {}
        }
        if (isnan(value)) {
            printf("Particle::getEnergy nan, value: %f, parameter: \n", value);
            while (true) {}
        }
        return value;
    }

    __device__ __inline__ void setEnergy(float value, int parameter)
    {
        if (value < 0) {
            printf("Particle::setEnergy negative, value: %f, parameter: %d\n", value, parameter);
            while (true) {}
        }
        if (isnan(value)) {
            printf("Particle::setEnergy nan, value: %f, parameter: %d\n", value, parameter);
            while (true) {}
        }
        atomicExch(&_energy, value);
    }

    __device__ __inline__ void changeEnergy(float changeValue, int parameter = -1)
    {
        auto const origValue = atomicAdd(&_energy, changeValue);
        if (getEnergy() < 0) {
            printf("Particle::changeEnergy negative, before: %f, after: %f, change: %f, parameter: %d\n", origValue, getEnergy(), changeValue, parameter);
            while (true) {}
        }
        if (isnan(getEnergy())) {
            printf("Particle::changeEnergy nan, before: %f, after: %f, change: %f, parameter: %d\n", origValue, getEnergy(), changeValue, parameter);
            while (true) {}
        }
    }


private:
    float _energy;
};