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
    float radiationMinCellEnergy = 500;
    int radiationMinCellAge = 500000;
    bool clusterDecay = true;
    float clusterDecayProb = 0.0001f;
    
    bool cellFunctionConstructionInheritColor = true;
    float cellFunctionConstructorOffspringDistance = 2.0f;
    float cellFunctionConstructorConnectingCellMaxDistance = 1.5f;
    float cellFunctionConstructorActivityThreshold = 0.25f;

    float cellFunctionAttackerRadius = 1.6f;
    float cellFunctionAttackerStrength = 0.05f;
    float cellFunctionAttackerEnergyDistributionRadius = 3.6f;
    bool cellFunctionAttackerEnergyDistributionSameColor = true;
    float cellFunctionAttackerEnergyDistributionValue = 10.0f;
    float cellFunctionAttackerInhomogeneityBonusFactor = 1.0f;
    float cellFunctionAttackerActivityThreshold = 0.25f;

    bool cellFunctionTransmitterEnergyDistributionSameColor = true;
    float cellFunctionTransmitterEnergyDistributionRadius = 3.6f;
    float cellFunctionTransmitterEnergyDistributionValue = 10.0f;

    float cellFunctionMuscleContractionExpansionDelta = 0.05f;
    float cellFunctionMuscleMovementDelta = 0.01f;
    float cellFunctionMuscleBendingAngle = 5.0f;

    float cellFunctionSensorRange = 255.0f;

    bool particleTransformationAllowed = false;
    bool particleTransformationRandomCellFunction = false;
    int particleTransformationMaxGenomeSize = 300;

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
            && cellFunctionAttackerEnergyDistributionValue == cellFunctionAttackerEnergyDistributionValue
            && cellFunctionAttackerActivityThreshold == other.cellFunctionAttackerActivityThreshold
            && cellFunctionMuscleContractionExpansionDelta == other.cellFunctionMuscleContractionExpansionDelta
            && cellFunctionMuscleMovementDelta == other.cellFunctionMuscleMovementDelta
            && cellFunctionMuscleBendingAngle == other.cellFunctionMuscleBendingAngle && innerFriction == other.innerFriction
            && particleTransformationMaxGenomeSize == other.particleTransformationMaxGenomeSize
            && cellFunctionConstructorOffspringDistance == other.cellFunctionConstructorOffspringDistance
            && cellFunctionConstructorConnectingCellMaxDistance == other.cellFunctionConstructorConnectingCellMaxDistance
            && cellFunctionConstructorActivityThreshold == other.cellFunctionConstructorActivityThreshold
            && cellFunctionTransmitterEnergyDistributionRadius == other.cellFunctionTransmitterEnergyDistributionRadius
            && cellFunctionTransmitterEnergyDistributionValue == other.cellFunctionTransmitterEnergyDistributionValue
            && cellFunctionAttackerEnergyDistributionSameColor == other.cellFunctionAttackerEnergyDistributionSameColor
            && cellFunctionTransmitterEnergyDistributionSameColor == other.cellFunctionTransmitterEnergyDistributionSameColor
            && particleTransformationAllowed == other.particleTransformationAllowed
            && particleTransformationRandomCellFunction == other.particleTransformationRandomCellFunction
            && particleTransformationMaxGenomeSize == other.particleTransformationMaxGenomeSize
            && cellFunctionConstructionInheritColor == other.cellFunctionConstructionInheritColor && clusterDecayProb == other.clusterDecayProb
            && clusterDecay == other.clusterDecay && radiationMinCellAge == other.radiationMinCellAge && radiationMinCellEnergy == other.radiationMinCellEnergy
        ;
    }

    bool operator!=(SimulationParameters const& other) const { return !operator==(other); }
};
