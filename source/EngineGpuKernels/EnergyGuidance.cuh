#pragma once

#include "EngineInterface/CellFunctionEnums.h"

#include "SimulationData.cuh"
#include "Cell.cuh"

class EnergyGuidance
{
public:
    static __inline__ __device__ void processing(SimulationData& data, Token * token)
    {
        auto cell = token->cell;
        uint8_t cmd = token->memory[EnergyGuidance_Input] % static_cast<int>(EnergyGuidanceIn_Count);
        float valueCell = static_cast<uint8_t>(token->memory[EnergyGuidance_InValueCell]);
        float valueToken = static_cast<uint8_t>(token->memory[EnergyGuidance_InValueToken]);
        const float amount = 10.0;

        auto cellMinEnergy = SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellMinEnergy, data, cell->absPos);

        if (EnergyGuidanceIn_Deactivated == cmd) {
            return;
        }

        if (EnergyGuidanceIn_BalanceCell == cmd) {
            if (cell->energy > (cellMinEnergy + valueCell + amount)) {
                cell->energy -= amount;
                token->energy += amount;
            }
            else if (token->energy > (cudaSimulationParameters.tokenMinEnergy + valueToken + amount)) {
                cell->energy += amount;
                token->energy -= amount;
            }
        }
        if (EnergyGuidanceIn_BalanceToken == cmd) {
            if (token->energy > (cudaSimulationParameters.tokenMinEnergy + valueToken + amount)) {
                cell->energy += amount;
                token->energy -= amount;
            } else if (cell->energy > (cellMinEnergy + valueCell + amount)) {
                cell->energy -= amount;
                token->energy += amount;
            }
        }
        if (EnergyGuidanceIn_BalanceBoth == cmd) {
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
        if (EnergyGuidanceIn_HarvestCell == cmd) {
            if (cell->energy > cellMinEnergy + valueCell + amount) {
                cell->energy -= amount;
                token->energy += amount;
            }
        }
        if (EnergyGuidanceIn_HarvestToken == cmd) {
            if (token->energy > cudaSimulationParameters.tokenMinEnergy + valueToken + amount) {
                cell->energy += amount;
                token->energy -= amount;
            }
        }
    }

};