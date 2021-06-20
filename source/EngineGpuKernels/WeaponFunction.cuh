#pragma once

#include "Cell.cuh"
#include "ConstantMemory.cuh"
#include "EngineInterface/ElementaryTypes.h"
#include "SimulationData.cuh"
#include "Token.cuh"

class WeaponFunction
{
public:
    __inline__ __device__ static void processing(Token* token, SimulationData* data);

private:
    __inline__ __device__ static bool isHomogene(Cell* cell);
    __inline__ __device__ static float calcOpenAngle(Cell* cell, float2 direction);
};

__inline__ __device__ void WeaponFunction::processing(Token* token, SimulationData* data)
{
    auto const& cell = token->cell;
    auto& tokenMem = token->memory;
    tokenMem[Enums::Weapon::OUTPUT] = Enums::WeaponOut::NO_TARGET;
    int const minMass = static_cast<unsigned char>(tokenMem[Enums::Weapon::IN_MIN_MASS]);
    int maxMass = static_cast<unsigned char>(tokenMem[Enums::Weapon::IN_MAX_MASS]);
    if (0 == maxMass) {
        maxMass = 16000;  //large value => no max mass check
    }

    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            auto const searchPos = float2{cell->absPos.x + x, cell->absPos.y + y};
            auto const otherCell = data->cellMap.get(searchPos);

            if (!otherCell) {
                continue;
            }
            if (otherCell->cluster == cell->cluster) {
                continue;
            }
            if (otherCell->cluster->numCellPointers < minMass || otherCell->cluster->numCellPointers > maxMass) {
                continue;
            }
            if (otherCell->tryLock()) {
                /*
                auto const mass = static_cast<float>(cell->cluster->numCellPointers);
                auto const otherMass = static_cast<float>(otherCell->cluster->numCellPointers);
                auto const energyToTransfer = / *sqrt* /(mass / otherMass)*(mass / otherMass)*cudaSimulationParameters.cellFunctionWeaponStrength;
*/
                auto energyToTransfer =
                    otherCell->getEnergy_safe() * cudaSimulationParameters.cellFunctionWeaponStrength + 1.0f;

                if (abs(cudaSimulationParameters.cellFunctionWeaponGeometryDeviationExponent) > FP_PRECISION) {
                    auto d = otherCell->absPos - cell->absPos;
                    auto angle1 = calcOpenAngle(cell, d);
                    auto angle2 = calcOpenAngle(otherCell, d * (-1));
                    auto deviation = 1.0f - abs(360.0f - (angle1 + angle2)) / 360.0f;   //1 = no deviation, 0 = max deviation
               
                    energyToTransfer = energyToTransfer
                        * powf(max(0.0f, min(1.0f, deviation)),
                               cudaSimulationParameters.cellFunctionWeaponGeometryDeviationExponent);
                }

                auto homogene = isHomogene(cell);
                auto otherHomogene = isHomogene(otherCell);
                if (!homogene && otherHomogene) {
                    energyToTransfer =
                        energyToTransfer * cudaSimulationParameters.cellFunctionWeaponInhomogeneousColorFactor;
                }
                if (homogene && otherHomogene && cell->metadata.color != otherCell->metadata.color) {
                    energyToTransfer =
                        energyToTransfer * cudaSimulationParameters.cellFunctionWeaponInhomogeneousColorFactor;
                }

                if (otherCell->getEnergy_safe() > energyToTransfer) {
                    otherCell->changeEnergy_safe(-energyToTransfer);
                    token->changeEnergy(energyToTransfer / 2.0f);
                    cell->changeEnergy_safe(energyToTransfer / 2.0f);
                    token->memory[Enums::Weapon::OUTPUT] = Enums::WeaponOut::STRIKE_SUCCESSFUL;
                }
                otherCell->cluster->unfreeze(30);
                otherCell->releaseLock();
            }
        }
    }
    if (cudaSimulationParameters.cellFunctionWeaponEnergyCost > 0) {
        auto const cellEnergy = cell->getEnergy_safe();
        auto& pos = cell->absPos;
        float2 particleVel = (cell->vel * cudaSimulationParameters.radiationVelocityMultiplier)
            + float2{
                (data->numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation,
                (data->numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation};
        float2 particlePos = pos + Math::normalized(particleVel) * 1.5f;
        data->cellMap.mapPosCorrection(particlePos);

        particlePos = particlePos - particleVel;  //because particle will still be moved in current time step
        auto const radiationEnergy = min(cellEnergy, cudaSimulationParameters.cellFunctionWeaponEnergyCost);
        cell->changeEnergy_safe(-radiationEnergy);
        EntityFactory factory;
        factory.init(data);
        auto particle = factory.createParticle(radiationEnergy, particlePos, particleVel, {cell->metadata.color});
    }
}

__inline__ __device__ bool WeaponFunction::isHomogene(Cell* cell)
{
    int color = cell->metadata.color;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto otherCell = cell->connections[i];
        if (color != otherCell->metadata.color) {
            return false;
        }
    }
    return true;
}

__inline__ __device__ float WeaponFunction::calcOpenAngle(Cell* cell, float2 direction)
{
    if (0 == cell->numConnections) {
        return 0.0f;
    }
    if (1 == cell->numConnections) {
        return 365.0f;
    }

    auto refAngle = Math::angleOfVector(direction);

    float largerAngle = Math::angleOfVector(cell->connections[0]->absPos - cell->absPos);
    float smallerAngle = largerAngle;

    for (int i = 1; i < cell->numConnections; ++i) {
        auto otherCell = cell->connections[i];
        auto angle = Math::angleOfVector(otherCell->absPos - cell->absPos);
        if (largerAngle >= refAngle) {
            if (largerAngle > angle && angle >= refAngle) {
                largerAngle = angle;
            }
        } else {
            if (largerAngle > angle || angle >= refAngle) {
                largerAngle = angle;
            }
        }

        if (smallerAngle <= refAngle) {
            if (smallerAngle < angle && angle <= refAngle) {
                smallerAngle = angle;
            }
        } else {
            if (smallerAngle < angle || angle <= refAngle) {
                smallerAngle = angle;
            }
        }
    }
    return Math::subtractAngle(largerAngle, smallerAngle);
}
