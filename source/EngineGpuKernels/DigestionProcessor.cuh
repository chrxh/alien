#pragma once

#include "EngineInterface/Enums.h"
#include "Cell.cuh"
#include "ConstantMemory.cuh"
#include "SimulationData.cuh"
#include "Token.cuh"

class DigestionProcessor
{
public:
    __inline__ __device__ static void process(Token* token, SimulationData& data, SimulationResult& result);

private:
    __inline__ __device__ static bool isHomogene(Cell* cell);
    __inline__ __device__ static float calcOpenAngle(Cell* cell, float2 direction);

    __inline__ __device__ static bool isConnectedConnected(Cell* cell, Cell* otherCell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void DigestionProcessor::process(Token* token, SimulationData& data, SimulationResult& result)
{
    auto const& cell = token->cell;
    auto& tokenMem = token->memory;
    tokenMem[Enums::Digestion_Output] = Enums::DigestionOut_NoTarget;

    if (cell->tryLock()) {

        auto color = calcMod(token->memory[Enums::Digestion_InColor], 7);

        Cell* otherCells[18];
        int numOtherCells;
        data.cellMap.get(otherCells, 18, numOtherCells, cell->absPos, 1.6f);
        for (int i = 0; i < numOtherCells; ++i) {
            Cell* otherCell = otherCells[i];
            if (otherCell->tryLock()) {
                if (!isConnectedConnected(cell, otherCell) && !otherCell->barrier) {
                    auto energyToTransfer = otherCell->energy * cudaSimulationParameters.cellFunctionWeaponStrength + 1.0f;

                    auto cellFunctionWeaponGeometryDeviationExponent =
                        SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionWeaponGeometryDeviationExponent, data, cell->absPos);

                    if (abs(cellFunctionWeaponGeometryDeviationExponent) > FP_PRECISION) {
                        auto d = otherCell->absPos - cell->absPos;
                        auto angle1 = calcOpenAngle(cell, d);
                        auto angle2 = calcOpenAngle(otherCell, d * (-1));
                        auto deviation = 1.0f - abs(360.0f - (angle1 + angle2)) / 360.0f;  //1 = no deviation, 0 = max deviation

                        energyToTransfer *= powf(max(0.0f, min(1.0f, deviation)), cellFunctionWeaponGeometryDeviationExponent);
                    }

                    auto otherCellColor = calcMod(otherCell->metadata.color, 7);
                    if (otherCellColor != color) {
                        auto cellFunctionWeaponColorTargetMismatchPenalty =
                            SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionWeaponColorTargetMismatchPenalty, data, cell->absPos);

                        energyToTransfer *= (1.0f - cellFunctionWeaponColorTargetMismatchPenalty);
                    }

                    auto cellFunctionWeaponColorPenalty =
                        SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionWeaponColorDominance, data, cell->absPos);

                    auto homogene = isHomogene(cell);
                    auto otherHomogene = isHomogene(otherCell);
                    if (!homogene /* && otherHomogene*/) {
                        energyToTransfer *= (1.0f - cellFunctionWeaponColorPenalty);
                    }
                    auto isColorSuperior = [](unsigned char color1, unsigned char color2) {
                        color1 = color1 % 7;
                        color2 = color2 % 7;
                        if (color1 == color2 + 1 || (color1 == 0 && color2 == 6)) {
                            return true;
                        }
                        return false;
                    };
                    if (homogene && otherHomogene && !isColorSuperior(cell->metadata.color, otherCell->metadata.color)) {
                        energyToTransfer *= (1.0f - cellFunctionWeaponColorPenalty);
                    }

                    auto cellFunctionWeaponConnectionsMismatchPenalty =
                        SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionWeaponConnectionsMismatchPenalty, data, cell->absPos);
                    if (otherCell->numConnections > cell->numConnections + 1) {
                        energyToTransfer *= (1 - cellFunctionWeaponConnectionsMismatchPenalty) * (1 - cellFunctionWeaponConnectionsMismatchPenalty);
                    }
                    if (otherCell->numConnections == cell->numConnections + 1) {
                        energyToTransfer *= (1 - cellFunctionWeaponConnectionsMismatchPenalty);
                    }
                    //tag = number of tokens on cell
                    if (otherCell->tag > 0) {
                        auto cellFunctionWeaponTokenPenalty =
                            SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionWeaponTokenPenalty, data, cell->absPos);
                        energyToTransfer *= (1.0f - cellFunctionWeaponTokenPenalty);
                    }

                    if (energyToTransfer >= 0) {
                        if (otherCell->energy > energyToTransfer) {
                            otherCell->energy -= energyToTransfer;
                            token->energy += energyToTransfer / 2;
                            cell->energy += energyToTransfer / 2;
                            if (token->memory[Enums::Digestion_Output] != Enums::DigestionOut_Poisoned) {
                                token->memory[Enums::Digestion_Output] = Enums::DigestionOut_Success;
                            }
                            result.incSuccessfulAttack();
                        }
                    } else {
                        auto cellMinEnergy = SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellMinEnergy, data, cell->absPos);

                        if (token->energy > -energyToTransfer / 2 + cudaSimulationParameters.tokenMinEnergy * 2
                            && cell->energy > -energyToTransfer / 2 + cellMinEnergy) {
                            otherCell->energy -= energyToTransfer;
                            token->energy += energyToTransfer / 2;
                            cell->energy += energyToTransfer / 2;
                            token->memory[Enums::Digestion_Output] = Enums::DigestionOut_Poisoned;
                        }
                    }
                }
                otherCell->releaseLock();
            }
        }
        if (Enums::DigestionOut_NoTarget == token->memory[Enums::Digestion_Output]) {
            result.incFailedAttack();
        }
        auto cellFunctionWeaponEnergyCost = SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionWeaponEnergyCost, data, cell->absPos);
        if (cellFunctionWeaponEnergyCost > 0) {
            auto const cellEnergy = cell->energy;
            auto& pos = cell->absPos;
            float2 particleVel = (cell->vel * cudaSimulationParameters.radiationVelocityMultiplier)
                + float2{
                    (data.numberGen1.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation,
                    (data.numberGen1.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation};
            float2 particlePos = pos + Math::normalized(particleVel) * 1.5f;
            data.cellMap.correctPosition(particlePos);

            particlePos = particlePos - particleVel;  //because particle will still be moved in current time step
            auto const radiationEnergy = min(cellEnergy, cellFunctionWeaponEnergyCost);
            cell->energy -= radiationEnergy;
            EntityFactory factory;
            factory.init(&data);
            auto particle = factory.createParticle(radiationEnergy, particlePos, particleVel, {cell->metadata.color});
        }
        cell->releaseLock();
    }
}

__inline__ __device__ bool DigestionProcessor::isHomogene(Cell* cell)
{
    int color = cell->metadata.color;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto otherCell = cell->connections[i].cell;
        if ((color % 7) != (otherCell->metadata.color % 7)) {
            return false;
        }
        for (int j = 0; j < otherCell->numConnections; ++j) {
            auto otherOtherCell = otherCell->connections[j].cell;
            if ((color % 7) != (otherOtherCell->metadata.color % 7)) {
                return false;
            }
        }
    }
    return true;
}

__inline__ __device__ float DigestionProcessor::calcOpenAngle(Cell* cell, float2 direction)
{
    if (0 == cell->numConnections) {
        return 0.0f;
    }
    if (1 == cell->numConnections) {
        return 365.0f;
    }

    auto refAngle = Math::angleOfVector(direction);

    float largerAngle = Math::angleOfVector(cell->connections[0].cell->absPos - cell->absPos);
    float smallerAngle = largerAngle;

    for (int i = 1; i < cell->numConnections; ++i) {
        auto otherCell = cell->connections[i].cell;
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

__inline__ __device__ bool DigestionProcessor::isConnectedConnected(Cell* cell, Cell* otherCell)
{
    if (cell == otherCell) {
        return true;
    }
    bool result = false;
    for (int i = 0; i < otherCell->numConnections; ++i) {
        auto const& connectedCell = otherCell->connections[i].cell;
        if (connectedCell == cell) {
            result = true;
            break;
        }

        for (int j = 0; j < connectedCell->numConnections; ++j) {
            auto const& connectedConnectedCell = connectedCell->connections[j].cell;
            if (connectedConnectedCell == cell) {
                result = true;
                break;
            }
        }
        if (result) {
            return true;
        }
    }
    return result;
}
