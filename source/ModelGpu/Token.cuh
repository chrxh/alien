#pragma once
#include "Base.cuh"
#include "Definitions.cuh"
#include "ConstantMemory.cuh"

struct Token
{
    char memory[MAX_TOKEN_MEM_SIZE];
    Cell* sourceCell;
    Cell* cell;

    __inline__ __device__ int getMaxConnectionsForConstructor()
    {
        return static_cast<unsigned char>(memory[Enums::Constr::IN_CELL_MAX_CONNECTIONS])
            % (cudaSimulationParameters.cellMaxBonds + 1);
    }

    __device__ __inline__ void setEnergy(float value)
    {
        if (value < 0) {
            printf("Token::setEnergy negative, value: %f\n", value);
            while (true) {}
        }
        if (isnan(value)) {
            printf("Token::setEnergy nan, value: %f\n", value);
            while (true) {}
        }
        _energy = value;
    }

    __device__ __inline__ void changeEnergy(float changeValue)
    {
        int origEnergy = _energy;
        _energy += changeValue;

        if (getEnergy() < 0) {
            printf(
                "Token::changeEnergy negative, before: %f, after: %f, change: %f\n",
                origEnergy, getEnergy(), changeValue);
            while (true) {}
        }
        if (isnan(getEnergy())) {
            printf(
                "Token::changeEnergy nan, before: %f, after: %f, change: %f\n",
                origEnergy, getEnergy(), changeValue);
            while (true) {}
        }

    }

    __device__ __inline__ float getEnergy() const
    {
        if (_energy < 0) {
            printf("Token::getEnergy negative, value: %f\n", _energy);
            while (true) {}
        }
        if (isnan(_energy)) {
            printf("Token::getEnergy nan, value: %f\n", _energy);
            while (true) {}
        }
        return _energy;
    }

private:
    float _energy;
};
