#pragma once

#include "SimulationParametersSpotValues.h"
#include "ParticleSource.h"
#include "FlowCenter.h"

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

    float radiationAbsorptionByCellColor[MAX_COLORS] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float radiationProb = 0.03f;
    float radiationVelocityMultiplier = 1.0f;
    float radiationVelocityPerturbation = 0.5f;
    float radiationMinCellEnergy = 500;
    int radiationMinCellAge = 500000;
    bool clusterDecay = true;
    float clusterDecayProb = 0.0001f;
    
    bool cellFunctionConstructionUnlimitedEnergy = false;
    bool cellFunctionConstructionInheritColor = true;
    float cellFunctionConstructorOffspringDistance = 2.0f;
    float cellFunctionConstructorConnectingCellMaxDistance = 1.5f;
    float cellFunctionConstructorActivityThreshold = 0.1f;

    float cellFunctionAttackerRadius = 1.6f;
    float cellFunctionAttackerStrength = 0.05f;
    float cellFunctionAttackerEnergyDistributionRadius = 3.6f;
    bool cellFunctionAttackerEnergyDistributionSameColor = true;
    float cellFunctionAttackerEnergyDistributionValue = 10.0f;
    float cellFunctionAttackerColorInhomogeneityFactor = 1.0f;
    float cellFunctionAttackerActivityThreshold = 0.1f;

    bool cellFunctionTransmitterEnergyDistributionSameColor = true;
    float cellFunctionTransmitterEnergyDistributionRadius = 3.6f;
    float cellFunctionTransmitterEnergyDistributionValue = 10.0f;

    float cellFunctionMuscleContractionExpansionDelta = 0.05f;
    float cellFunctionMuscleMovementAcceleration = 0.02f;
    float cellFunctionMuscleBendingAngle = 5.0f;

    float cellFunctionSensorRange = 255.0f;
    float cellFunctionSensorActivityThreshold = 0.1f;

    bool particleTransformationAllowed = false;
    bool particleTransformationRandomCellFunction = false;
    int particleTransformationMaxGenomeSize = 300;

    //particle sources
    int numParticleSources = 0;
    ParticleSource particleSources[MAX_PARTICLE_SOURCES];

    //flow centers
    int numFlowCenters = 0;
    FlowCenter flowCenters[MAX_FLOW_CENTERS];

    //inherit color
    bool operator==(SimulationParameters const& other) const
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (radiationAbsorptionByCellColor[i] != other.radiationAbsorptionByCellColor[i]) {
                return false;
            }
        }
        if (numParticleSources != other.numParticleSources) {
            return false;
        }
        for (int i = 0; i < numParticleSources; ++i) {
            if (particleSources[i] != other.particleSources[i]) {
                return false;
            }
        }
        if (numFlowCenters != other.numFlowCenters) {
            return false;
        }
        for (int i = 0; i < 2; ++i) {
            if (flowCenters[i] != other.flowCenters[i]) {
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
            && cellFunctionAttackerColorInhomogeneityFactor == other.cellFunctionAttackerColorInhomogeneityFactor
            && cellFunctionAttackerRadius == other.cellFunctionAttackerRadius
            && cellFunctionAttackerEnergyDistributionRadius == other.cellFunctionAttackerEnergyDistributionRadius
            && cellFunctionAttackerEnergyDistributionValue == cellFunctionAttackerEnergyDistributionValue
            && cellFunctionAttackerActivityThreshold == other.cellFunctionAttackerActivityThreshold
            && cellFunctionMuscleContractionExpansionDelta == other.cellFunctionMuscleContractionExpansionDelta
            && cellFunctionMuscleMovementAcceleration == other.cellFunctionMuscleMovementAcceleration
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
            && cellFunctionSensorActivityThreshold == other.cellFunctionSensorActivityThreshold
            && cellFunctionConstructionUnlimitedEnergy == other.cellFunctionConstructionUnlimitedEnergy
        ;
    }

    bool operator!=(SimulationParameters const& other) const { return !operator==(other); }
};
