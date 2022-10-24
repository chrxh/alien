#pragma once

#include "SimulationParametersSpotValues.h"

struct SimulationParameters
{
    SimulationParametersSpotValues spotValues;

    float timestepSize = 1.0f;            
    float cellMaxVel = 2.0f;              
    float cellMaxBindingDistance = 2.6f;
    float cellRepulsionStrength = 0.08f;  

    float cellMinDistance = 0.3f;         
    float cellMaxCollisionDistance = 1.3f;
    float cellMaxForceDecayProb = 0.2f;
    int cellMaxBonds = 6;
    int cellMaxExecutionOrderNumber = 6;
    int cellCreationTokenAccessNumber = 0;

    float cellFunctionWeaponStrength = 0.1f;
    float cellFunctionSensorRange = 255.0f;

    float radiationProb = 0.03f;
    float radiationVelocityMultiplier = 1.0f;
    float radiationVelocityPerturbation = 0.5f;

    bool operator==(SimulationParameters const& other) const
    {
        return spotValues == other.spotValues && timestepSize == other.timestepSize && cellMaxVel == other.cellMaxVel
            && cellMaxBindingDistance == other.cellMaxBindingDistance && cellMinDistance == other.cellMinDistance
            && cellMaxCollisionDistance == other.cellMaxCollisionDistance && cellMaxForceDecayProb == other.cellMaxForceDecayProb
            && cellMaxBonds == other.cellMaxBonds && cellMaxExecutionOrderNumber == other.cellMaxExecutionOrderNumber
            && cellCreationTokenAccessNumber == other.cellCreationTokenAccessNumber && cellFunctionWeaponStrength == other.cellFunctionWeaponStrength
            && cellFunctionSensorRange == other.cellFunctionSensorRange && radiationProb == other.radiationProb
            && radiationVelocityMultiplier == other.radiationVelocityMultiplier && radiationVelocityPerturbation == other.radiationVelocityPerturbation
            && cellRepulsionStrength == other.cellRepulsionStrength;
    }

    bool operator!=(SimulationParameters const& other) const { return !operator==(other); }
};
