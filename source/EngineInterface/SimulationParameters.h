#pragma once

#include "SimulationParametersSpotValues.h"

struct SimulationParameters
{
    SimulationParametersSpotValues spotValues;

    float timestepSize = 1.0f;
    float innerFriction = 0.3f;
    float cellMaxVelocity = 2.0f;              
    float cellMaxBindingDistance = 3.6f;
    float cellRepulsionStrength = 0.08f;

    float cellNormalEnergy = 100.0f;
    float cellMinDistance = 0.3f;         
    float cellMaxCollisionDistance = 1.3f;
    float cellMaxForceDecayProb = 0.2f;
    int cellMaxBonds = 6;
    int cellMaxExecutionOrderNumbers = 6;

    bool radiationAbsorptionByCellColor[MAX_COLORS] = {false, false, true, false, false, false, false};
    float radiationProb = 0.03f;
    float radiationVelocityMultiplier = 1.0f;
    float radiationVelocityPerturbation = 0.5f;
    float radiationMinEnergy = 500;
    int radiationMinAge = 500000;
    bool clusterDecay = true;
    float clusterDecayProb = 0.0001f;
    
    bool cellFunctionConstructionInheritColor = true;
    float cellFunctionConstructorOffspringCellDistance = 2.0f;
    float cellFunctionConstructorConnectingCellDistance = 1.5f;
    float cellFunctionConstructorActivityThreshold = 0.25f;

    float cellFunctionAttackerRadius = 1.6f;
    float cellFunctionAttackerStrength = 0.05f;
    float cellFunctionAttackerEnergyDistributionRadius = 3.6f;
    bool cellFunctionAttackerDistributeEnergySameColor = true;
    float cellFunctionAttackerDistributeEnergy = 10.0f;
    float cellFunctionAttackerInhomogeneityBonusFactor = 1.0f;
    float cellFunctionAttackerActivityThreshold = 0.25f;

    bool cellFunctionTransmitterDistributeEnergySameColor = true;
    float cellFunctionTransmitterEnergyDistributionRadius = 3.6f;
    float cellFunctionTransmitterDistributeEnergy = 10.0f;

    float cellFunctionMuscleActivityThreshold = 0.25f;
    float cellFunctionMuscleOppositeActivityThreshold = -0.25f;
    float cellFunctionMuscleContractionExpansionDelta = 0.05f;
    float cellFunctionMuscleMovementDelta = 0.01f;
    float cellFunctionMuscleBendingAngle = 5.0f;

    float cellFunctionSensorRange = 255.0f;

    bool particleAllowTransformationToCell = false;
    bool particleTransformationRandomCellFunction = false;
    int randomMaxGenomeSize = 300;

    //inherit color
    bool operator==(SimulationParameters const& other) const
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (radiationAbsorptionByCellColor[i] != other.radiationAbsorptionByCellColor[i]) {
                return false;
            }
        }
        return spotValues == other.spotValues && timestepSize == other.timestepSize && cellMaxVelocity == other.cellMaxVelocity
            && cellMaxBindingDistance == other.cellMaxBindingDistance && cellMinDistance == other.cellMinDistance
            && cellMaxCollisionDistance == other.cellMaxCollisionDistance && cellMaxForceDecayProb == other.cellMaxForceDecayProb
            && cellMaxBonds == other.cellMaxBonds && cellMaxExecutionOrderNumbers == other.cellMaxExecutionOrderNumbers
            && cellFunctionAttackerStrength == other.cellFunctionAttackerStrength
            && cellFunctionSensorRange == other.cellFunctionSensorRange && radiationProb == other.radiationProb
            && radiationVelocityMultiplier == other.radiationVelocityMultiplier && radiationVelocityPerturbation == other.radiationVelocityPerturbation
            && cellRepulsionStrength == other.cellRepulsionStrength && cellNormalEnergy == other.cellNormalEnergy
            && cellFunctionAttackerInhomogeneityBonusFactor == other.cellFunctionAttackerInhomogeneityBonusFactor
            && cellFunctionAttackerRadius == other.cellFunctionAttackerRadius
            && cellFunctionAttackerEnergyDistributionRadius == other.cellFunctionAttackerEnergyDistributionRadius
            && cellFunctionAttackerDistributeEnergy == cellFunctionAttackerDistributeEnergy
            && cellFunctionAttackerActivityThreshold == other.cellFunctionAttackerActivityThreshold
            && cellFunctionMuscleActivityThreshold == other.cellFunctionMuscleActivityThreshold
            && cellFunctionMuscleOppositeActivityThreshold == other.cellFunctionMuscleOppositeActivityThreshold
            && cellFunctionMuscleContractionExpansionDelta == other.cellFunctionMuscleContractionExpansionDelta
            && cellFunctionMuscleMovementDelta == other.cellFunctionMuscleMovementDelta
            && cellFunctionMuscleBendingAngle == other.cellFunctionMuscleBendingAngle && innerFriction == other.innerFriction
            && particleTransformationRandomCellFunction == other.particleTransformationRandomCellFunction
            && randomMaxGenomeSize == other.randomMaxGenomeSize
            && cellFunctionConstructorOffspringCellDistance == other.cellFunctionConstructorOffspringCellDistance
            && cellFunctionConstructorConnectingCellDistance == other.cellFunctionConstructorConnectingCellDistance
            && cellFunctionConstructorActivityThreshold == other.cellFunctionConstructorActivityThreshold
            && cellFunctionTransmitterEnergyDistributionRadius == other.cellFunctionTransmitterEnergyDistributionRadius
            && cellFunctionTransmitterDistributeEnergy == other.cellFunctionTransmitterDistributeEnergy
            && cellFunctionAttackerDistributeEnergySameColor == other.cellFunctionAttackerDistributeEnergySameColor
            && cellFunctionTransmitterDistributeEnergySameColor == other.cellFunctionTransmitterDistributeEnergySameColor
        ;
    }

    bool operator!=(SimulationParameters const& other) const { return !operator==(other); }
};
