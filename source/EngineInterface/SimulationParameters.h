#pragma once

#include "SimulationParametersSpotValues.h"

struct SimulationParameters
{
    SimulationParametersSpotValues spotValues;

    float timestepSize = 1.0f;
    float innerfriction = 0.3f;
    float cellMaxVelocity = 2.0f;              
    float cellMaxBindingDistance = 3.3f;
    float cellRepulsionStrength = 0.08f;

    float cellNormalEnergy = 100.0f;
    float cellMinDistance = 0.3f;         
    float cellMaxCollisionDistance = 1.3f;
    float cellMaxForceDecayProb = 0.2f;
    int cellMaxBonds = 6;
    int cellMaxExecutionOrderNumbers = 6;
    int cellCreationTokenAccessNumber = 0;

    float cellFunctionAttackerRadius = 1.6f;
    float cellFunctionAttackerStrength = 0.05f;
    float cellFunctionAttackerEnergyDistributionRadius = 3.6f;
    float cellFunctionAttackerInhomogeneityBonus = 2.0f;
    float cellFunctionAttackerActivityThreshold = 0.25f;
    float cellFunctionAttackerOutputPoisoned = -1;
    float cellFunctionAttackerOutputNothingFound = 0;
    float cellFunctionAttackerOutputSuccess = 1;

    float cellFunctionTransmitterEnergyDistributionRadius = 3.6f;

    float cellFunctionMuscleActivityThreshold = 0.25f;
    float cellFunctionMuscleOppositeActivityThreshold = -0.25f;
    float cellFunctionMuscleContractionExpansionDelta = 0.05f;
    float cellFunctionMuscleMovementDelta = 0.01f;
    float cellFunctionMuscleBendingAngle = 5.0f;
    
    float cellFunctionSensorRange = 255.0f;

    float radiationProb = 0.03f;
    float radiationVelocityMultiplier = 1.0f;
    float radiationVelocityPerturbation = 0.5f;

    bool cellFunctinoAllowCreateRandom = false;
    int cellFunctionRandomMaxGenomeSize = 100;

    //inherit color
    bool operator==(SimulationParameters const& other) const
    {
        return spotValues == other.spotValues && timestepSize == other.timestepSize && cellMaxVelocity == other.cellMaxVelocity
            && cellMaxBindingDistance == other.cellMaxBindingDistance && cellMinDistance == other.cellMinDistance
            && cellMaxCollisionDistance == other.cellMaxCollisionDistance && cellMaxForceDecayProb == other.cellMaxForceDecayProb
            && cellMaxBonds == other.cellMaxBonds && cellMaxExecutionOrderNumbers == other.cellMaxExecutionOrderNumbers
            && cellCreationTokenAccessNumber == other.cellCreationTokenAccessNumber
            && cellFunctionAttackerStrength == other.cellFunctionAttackerStrength
            && cellFunctionSensorRange == other.cellFunctionSensorRange && radiationProb == other.radiationProb
            && radiationVelocityMultiplier == other.radiationVelocityMultiplier && radiationVelocityPerturbation == other.radiationVelocityPerturbation
            && cellRepulsionStrength == other.cellRepulsionStrength && cellNormalEnergy == other.cellNormalEnergy
            && cellFunctionAttackerInhomogeneityBonus == other.cellFunctionAttackerInhomogeneityBonus
            && cellFunctionAttackerRadius == other.cellFunctionAttackerRadius
            && cellFunctionAttackerEnergyDistributionRadius == other.cellFunctionAttackerEnergyDistributionRadius
            && cellFunctionAttackerActivityThreshold == other.cellFunctionAttackerActivityThreshold
            && cellFunctionAttackerOutputPoisoned == other.cellFunctionAttackerOutputPoisoned
            && cellFunctionAttackerOutputNothingFound == other.cellFunctionAttackerOutputNothingFound
            && cellFunctionAttackerOutputSuccess == other.cellFunctionAttackerOutputSuccess
            && cellFunctionMuscleActivityThreshold == other.cellFunctionMuscleActivityThreshold
            && cellFunctionMuscleOppositeActivityThreshold == other.cellFunctionMuscleOppositeActivityThreshold
            && cellFunctionMuscleContractionExpansionDelta == other.cellFunctionMuscleContractionExpansionDelta
            && cellFunctionMuscleMovementDelta == other.cellFunctionMuscleMovementDelta
            && cellFunctionMuscleBendingAngle == other.cellFunctionMuscleBendingAngle && innerfriction == other.innerfriction
            && cellFunctinoAllowCreateRandom == other.cellFunctinoAllowCreateRandom && cellFunctionRandomMaxGenomeSize == other.cellFunctionRandomMaxGenomeSize
        ;
    }

    bool operator!=(SimulationParameters const& other) const { return !operator==(other); }
};
