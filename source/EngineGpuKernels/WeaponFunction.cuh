#pragma once

#include "EngineInterface/ElementaryTypes.h"

#include "SimulationData.cuh"
#include "Token.cuh"
#include "Cell.cuh"
#include "ConstantMemory.cuh"

class WeaponFunction
{
public:
    __inline__ __device__ static void processing(Token* token, SimulationData* data);

private:

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

    for (auto x = -2; x <= 2; ++x) {
        for (auto y = -2; y <= 2; ++y) {
            auto const searchPos = float2{ cell->absPos.x + x, cell->absPos.y + y };
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
                auto const energyToTransfer =
                    otherCell->getEnergy_safe() * cudaSimulationParameters.cellFunctionWeaponStrength + 1.0f;
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
        auto &pos = cell->absPos;
        float2 particleVel = (cell->vel * cudaSimulationParameters.radiationVelocityMultiplier)
            + float2{ (data->numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation,
            (data->numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation };
        float2 particlePos = pos + Math::normalized(particleVel) * 1.5f;
        data->cellMap.mapPosCorrection(particlePos);

        particlePos = particlePos - particleVel;	//because particle will still be moved in current time step
        auto const radiationEnergy = min(cellEnergy, cudaSimulationParameters.cellFunctionWeaponEnergyCost);
        cell->changeEnergy_safe(-radiationEnergy);
        EntityFactory factory;
        factory.init(data);
        auto particle = factory.createParticle(radiationEnergy, particlePos, particleVel, { cell->metadata.color });
    }
}

