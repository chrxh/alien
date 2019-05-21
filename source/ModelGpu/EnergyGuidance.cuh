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
            if (cell->energy > (cudaSimulationParameters.cellMinEnergy + valueCell + amount)) {
                cell->energy -= amount;
                token->energy += amount;
            }
            else if (token->energy > (cudaSimulationParameters.tokenMinEnergy + valueToken + amount)) {
                cell->energy += amount;
                token->energy -= amount;
            }
        }
        if (Enums::EnergyGuidanceIn::BALANCE_TOKEN == cmd) {
            if (token->energy > (cudaSimulationParameters.tokenMinEnergy + valueToken + amount)) {
                cell->energy += amount;
                token->energy -= amount;
            }
            else if (cell->energy > (cudaSimulationParameters.cellMinEnergy + valueCell + amount)) {
                cell->energy -= amount;
                token->energy += amount;
            }
        }
        if (Enums::EnergyGuidanceIn::BALANCE_BOTH == cmd) {
            if (token->energy > cudaSimulationParameters.tokenMinEnergy + valueToken + amount
                && cell->energy < cudaSimulationParameters.cellMinEnergy + valueCell) {
                cell->energy += amount;
                token->energy -= amount;
            }
            if (token->energy < cudaSimulationParameters.tokenMinEnergy + valueToken
                && cell->energy > cudaSimulationParameters.cellMinEnergy + valueCell + amount) {
                cell->energy -= amount;
                token->energy += amount;
            }
        }
        if (Enums::EnergyGuidanceIn::HARVEST_CELL == cmd) {
            if (cell->energy > cudaSimulationParameters.cellMinEnergy + valueCell + amount) {
                cell->energy -= amount;
                token->energy += amount;
            }
        }
        if (Enums::EnergyGuidanceIn::HARVEST_TOKEN == cmd) {
            if (token->energy > cudaSimulationParameters.tokenMinEnergy + valueToken + amount) {
                cell->energy += amount;
                token->energy -= amount;
            }
        }
    }

};