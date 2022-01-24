#pragma once

#include "EngineInterface/ElementaryTypes.h"

#include "SimulationData.cuh"
#include "Token.cuh"
#include "Cell.cuh"

class EnergyGuidance
{
public:
    static __inline__ __device__ void processing(SimulationData& data, Token * token)
    {
        auto cell = token->cell;
        uint8_t cmd = token->memory[Enums::EnergyGuidance_Input] % static_cast<int>(Enums::EnergyGuidanceIn_Count);
        float valueCell = static_cast<uint8_t>(token->memory[Enums::EnergyGuidance_InValueCell]);
        float valueToken = static_cast<uint8_t>(token->memory[Enums::EnergyGuidance_InValueToken]);
        const float amount = 10.0;

        auto cellMinEnergy = SpotCalculator::calc(&SimulationParametersSpotValues::cellMinEnergy, data, cell->absPos);

        if (Enums::EnergyGuidanceIn_Deactivated == cmd) {
            return;
        }

        if (Enums::EnergyGuidanceIn_BalanceCell == cmd) {
            if (cell->energy > (cellMinEnergy + valueCell + amount)) {
                cell->energy -= amount;
                token->energy += amount;
            }
            else if (token->energy > (cudaSimulationParameters.tokenMinEnergy + valueToken + amount)) {
                cell->energy += amount;
                token->energy -= amount;
            }
        }
        if (Enums::EnergyGuidanceIn_BalanceToken == cmd) {
            if (token->energy > (cudaSimulationParameters.tokenMinEnergy + valueToken + amount)) {
                cell->energy += amount;
                token->energy -= amount;
            } else if (cell->energy > (cellMinEnergy + valueCell + amount)) {
                cell->energy -= amount;
                token->energy += amount;
            }
        }
        if (Enums::EnergyGuidanceIn_BalanceBoth == cmd) {
            if (token->energy > cudaSimulationParameters.tokenMinEnergy + valueToken + amount
                && cell->energy < cellMinEnergy + valueCell) {
                cell->energy += amount;
                token->energy -= amount;
            }
            if (token->energy < cudaSimulationParameters.tokenMinEnergy + valueToken
                && cell->energy > cellMinEnergy + valueCell + amount) {
                cell->energy -= amount;
                token->energy += amount;
            }
        }
        if (Enums::EnergyGuidanceIn_HarvestCell == cmd) {
            if (cell->energy > cellMinEnergy + valueCell + amount) {
                cell->energy -= amount;
                token->energy += amount;
            }
        }
        if (Enums::EnergyGuidanceIn_HarvestToken == cmd) {
            if (token->energy > cudaSimulationParameters.tokenMinEnergy + valueToken + amount) {
                cell->energy += amount;
                token->energy -= amount;
            }
        }
    }

};