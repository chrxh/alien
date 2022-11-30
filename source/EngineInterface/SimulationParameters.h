#pragma once

#include "SimulationParametersSpotValues.h"

struct SimulationParameters
{
    SimulationParametersSpotValues spotValues;

    float timestepSize = 1.0f;
    float innerFriction = 0.3f;
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

    bool cellFunctionConstructionInheritColor = false;
    float cellFunctionConstructorOffspringCellDistance = 2.0f;
    float cellFunctionConstructorConnectingCellDistance = 1.6f;
    float cellFunctionConstructorActivityThreshold = 0.25f;

    float cellFunctionAttackerRadius = 1.6f;
    float cellFunctionAttackerStrength = 0.05f;
    float cellFunctionAttackerEnergyDistributionRadius = 3.6f;
    float cellFunctionAttackerInhomogeneityBonus = 1.0f;
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

    bool createRandomCellFunction = false;
    int randomMaxGenomeSize = 300;

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
            && cellFunctionMuscleBendingAngle == other.cellFunctionMuscleBendingAngle && innerFriction == other.innerFriction
            && createRandomCellFunction == other.createRandomCellFunction
            && randomMaxGenomeSize == other.randomMaxGenomeSize
            && cellFunctionConstructorOffspringCellDistance == other.cellFunctionConstructorOffspringCellDistance
            && cellFunctionConstructorConnectingCellDistance == other.cellFunctionConstructorConnectingCellDistance
            && cellFunctionConstructorActivityThreshold == other.cellFunctionConstructorActivityThreshold;
    }

    bool operator!=(SimulationParameters const& other) const { return !operator==(other); }
};
