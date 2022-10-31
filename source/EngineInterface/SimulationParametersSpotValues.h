#pragma once

#include "Constants.h"

struct SimulationParametersSpotValues
{
    float friction = 0.001f;
    float rigidity = 0.0f;
    float radiationFactor = 0.0002f;
    float cellMaxForce = 0.8f;
    float cellMinEnergy = 50.0f;

    float cellBindingForce = 1.0f;
    float cellFusionVelocity = 0.4f;
    float cellMaxBindingEnergy = 500000.0f;

    int cellColorTransitionDuration[MAX_COLORS] = {0, 0, 0, 0, 0, 0, 0};
    int cellColorTransitionTargetColor[MAX_COLORS] = {0, 1, 2, 3, 4, 5, 6};
    float cellFunctionWeaponEnergyCost = 0.2f;
    float cellFunctionWeaponFoodChainColorMatrix[MAX_COLORS][MAX_COLORS] = {
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1}};
    float cellFunctionWeaponGeometryDeviationExponent = 0.0f;
    float cellFunctionWeaponConnectionsMismatchPenalty = 0.33f;

    bool operator==(SimulationParametersSpotValues const& other) const
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                if (cellFunctionWeaponFoodChainColorMatrix[i][j] != other.cellFunctionWeaponFoodChainColorMatrix[i][j]) {
                    return false;
                }
            }
            if (cellColorTransitionDuration[i] != other.cellColorTransitionDuration[i]) {
                return false;
            }
            if (cellColorTransitionTargetColor[i] != other.cellColorTransitionTargetColor[i]) {
                return false;
            }
        }
        return friction == other.friction && rigidity == other.rigidity && radiationFactor == other.radiationFactor && cellMaxForce == other.cellMaxForce
            && cellMinEnergy == other.cellMinEnergy && cellBindingForce == other.cellBindingForce && cellFusionVelocity == other.cellFusionVelocity
            && cellFunctionWeaponEnergyCost == other.cellFunctionWeaponEnergyCost
            && cellFunctionWeaponGeometryDeviationExponent == other.cellFunctionWeaponGeometryDeviationExponent
            && cellMaxBindingEnergy == other.cellMaxBindingEnergy
            && cellFunctionWeaponConnectionsMismatchPenalty == other.cellFunctionWeaponConnectionsMismatchPenalty;
    }
};
