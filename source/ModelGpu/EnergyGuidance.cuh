#pragma once

#include "SimulationData.cuh"

class EnergyGuidance
{
public:
    static __inline__ __device__ void processing(Token * token)
    {
        auto cell = token->cell;
        uint8_t cmd = token->memory[Enums::EnergyGuidance::IN] % static_cast<int>(Enums::EnergyGuidanceIn::_COUNTER);
        float valueCell = static_cast<uint8_t>(token->memory[Enums::EnergyGuidance::IN_VALUE_CELL]);
        float valueToken = static_cast<uint8_t>(token->memory[Enums::EnergyGuidance::IN_VALUE_TOKEN]);
        const float amount = 10.0;

        if (Enums::EnergyGuidanceIn::DEACTIVATED == cmd) {
            return;
        }

        if (Enums::EnergyGuidanceIn::BALANCE_CELL == cmd) {
            if (cell->getEnergy() > (cudaSimulationParameters.cellMinEnergy + valueCell + amount)) {
                cell->changeEnergy(-amount, 4);
                token->changeEnergy(amount);
            }
            else if (token->getEnergy() > (cudaSimulationParameters.tokenMinEnergy + valueToken + amount)) {
                cell->changeEnergy(amount, 5);
                token->changeEnergy(-amount);
            }
        }
        if (Enums::EnergyGuidanceIn::BALANCE_TOKEN == cmd) {
            if (token->getEnergy() > (cudaSimulationParameters.tokenMinEnergy + valueToken + amount)) {
                cell->changeEnergy(amount, 6);
                token->changeEnergy(-amount);
            }
            else if (cell->getEnergy() > (cudaSimulationParameters.cellMinEnergy + valueCell + amount)) {
                cell->changeEnergy(-amount, 7);
                token->changeEnergy(amount);
            }
        }
        if (Enums::EnergyGuidanceIn::BALANCE_BOTH == cmd) {
            if (token->getEnergy() > cudaSimulationParameters.tokenMinEnergy + valueToken + amount
                && cell->getEnergy() < cudaSimulationParameters.cellMinEnergy + valueCell) {
                cell->changeEnergy(amount, 8);
                token->changeEnergy(-amount);
            }
            if (token->getEnergy() < cudaSimulationParameters.tokenMinEnergy + valueToken
                && cell->getEnergy() > cudaSimulationParameters.cellMinEnergy + valueCell + amount) {
                cell->changeEnergy(-amount, 9);
                token->changeEnergy(amount);
            }
        }
        if (Enums::EnergyGuidanceIn::HARVEST_CELL == cmd) {
            if (cell->getEnergy() > cudaSimulationParameters.cellMinEnergy + valueCell + amount) {
                cell->changeEnergy(-amount, 10);
                token->changeEnergy(amount);
            }
        }
        if (Enums::EnergyGuidanceIn::HARVEST_TOKEN == cmd) {
            if (token->getEnergy() > cudaSimulationParameters.tokenMinEnergy + valueToken + amount) {
                cell->changeEnergy(amount, 11);
                token->changeEnergy(-amount);
            }
        }
    }

};