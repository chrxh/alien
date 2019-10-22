#pragma once

#include "ModelBasic/ElementaryTypes.h"

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
    token->memory[Enums::Weapon::OUT] = Enums::WeaponOut::NO_TARGET;

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
            if (otherCell->tryLock()) {
                auto const energyToTransfer =
                    otherCell->getEnergy() * cudaSimulationParameters.cellFunctionWeaponStrength + 1.0f;
                if (otherCell->getEnergy() > energyToTransfer) {
                    otherCell->changeEnergy(-energyToTransfer);
                    token->changeEnergy(energyToTransfer / 2.0f);
                    cell->changeEnergy(energyToTransfer / 2.0f);
                    token->memory[Enums::Weapon::OUT] = Enums::WeaponOut::STRIKE_SUCCESSFUL;
                }
                otherCell->releaseLock();
            }
        }
    }
}

