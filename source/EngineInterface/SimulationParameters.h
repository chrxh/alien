#pragma once

#include <cstdint>

#include "SimulationParametersSpotValues.h"
#include "RadiationSource.h"
#include "SimulationParametersSpot.h"

struct FluidMotion
{
    float smoothingLength = 0.66f;
    float viscosityForce = 0.1f;
    float pressureForce = 0.1f;

    bool operator==(FluidMotion const& other) const
    {
        return smoothingLength == other.smoothingLength && viscosityForce == other.viscosityForce && pressureForce == other.pressureForce;
    }
    bool operator!=(FluidMotion const& other) const { return !operator==(other); }
};

struct CollisionMotion
{
    float cellMaxCollisionDistance = 1.3f;
    float cellRepulsionStrength = 0.08f;

    bool operator==(CollisionMotion const& other) const
    {
        return cellMaxCollisionDistance == other.cellMaxCollisionDistance && cellRepulsionStrength == other.cellRepulsionStrength;
    }
    bool operator!=(CollisionMotion const& other) const { return !operator==(other); }
};

union MotionData
{
    FluidMotion fluidMotion;
    CollisionMotion collisionMotion;
};

using MotionType = int;
enum MotionType_
{
    MotionType_Fluid,
    MotionType_Collision
};

struct SimulationParameters
{
    SimulationParametersSpotValues baseValues;

    uint32_t backgroundColor = 0x1b0000;

    float timestepSize = 1.0f;
    MotionType motionType = MotionType_Fluid;
    MotionData motionData = {FluidMotion()};

    float innerFriction = 0.3f;
    float cellMaxVelocity = 2.0f;              
    float cellMaxBindingDistance = 3.6f;

    float cellNormalEnergy = 100.0f;
    float cellMinDistance = 0.3f;         
    float cellMaxForceDecayProb = 0.2f;
    int cellMaxBonds = 6;
    int cellMaxExecutionOrderNumbers = 6;

    float radiationAbsorptionByCellColor[MAX_COLORS] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float radiationProb = 0.03f;
    float radiationVelocityMultiplier = 1.0f;
    float radiationVelocityPerturbation = 0.5f;
    float radiationMinCellEnergy = 500;
    int radiationMinCellAge = 500000;
    bool clusterDecay = false;
    float clusterDecayProb = 0.0001f;
    
    bool cellFunctionConstructionUnlimitedEnergy = false;
    float cellFunctionConstructorOffspringDistance = 2.0f;
    float cellFunctionConstructorConnectingCellMaxDistance = 1.5f;
    float cellFunctionConstructorActivityThreshold = 0.1f;
    bool cellFunctionConstructorMutationColor = false;
    bool cellFunctionConstructorMutationSelfReplication = false;

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
    float cellFunctionMuscleBendingAcceleration = 0.08f;
    float cellFunctionMuscleBendingAccelerationThreshold = 0.1f;

    float cellFunctionSensorRange = 255.0f;
    float cellFunctionSensorActivityThreshold = 0.1f;

    bool particleTransformationAllowed = false;
    bool particleTransformationRandomCellFunction = false;
    int particleTransformationMaxGenomeSize = 300;

    //particle sources
    int numParticleSources = 0;
    RadiationSource particleSources[MAX_PARTICLE_SOURCES];

    //spots
    int numSpots = 0;
    SimulationParametersSpot spots[MAX_SPOTS];

    //inherit color
    bool operator==(SimulationParameters const& other) const
    {
        if (motionType != other.motionType) {
            return false;
        }
        if (motionType == MotionType_Fluid) {
            if (motionData.fluidMotion != other.motionData.fluidMotion) {
                return false;
            }
        }
        if (motionType == MotionType_Collision) {
            if (motionData.collisionMotion != other.motionData.collisionMotion) {
                return false;
            }
        }

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
        if (numSpots != other.numSpots) {
            return false;
        }
        for (int i = 0; i < numSpots; ++i) {
            if (spots[i] != other.spots[i]) {
                return false;
            }
        }

        return backgroundColor == other.backgroundColor && baseValues == other.baseValues && timestepSize == other.timestepSize
            && cellMaxVelocity == other.cellMaxVelocity && cellMaxBindingDistance == other.cellMaxBindingDistance && cellMinDistance == other.cellMinDistance
            && cellMaxForceDecayProb == other.cellMaxForceDecayProb
            && cellMaxBonds == other.cellMaxBonds && cellMaxExecutionOrderNumbers == other.cellMaxExecutionOrderNumbers
            && cellFunctionAttackerStrength == other.cellFunctionAttackerStrength && cellFunctionSensorRange == other.cellFunctionSensorRange
            && radiationProb == other.radiationProb && radiationVelocityMultiplier == other.radiationVelocityMultiplier
            && radiationVelocityPerturbation == other.radiationVelocityPerturbation
            && cellNormalEnergy == other.cellNormalEnergy && cellFunctionAttackerColorInhomogeneityFactor == other.cellFunctionAttackerColorInhomogeneityFactor
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
            && clusterDecayProb == other.clusterDecayProb
            && clusterDecay == other.clusterDecay && radiationMinCellAge == other.radiationMinCellAge && radiationMinCellEnergy == other.radiationMinCellEnergy
            && cellFunctionSensorActivityThreshold == other.cellFunctionSensorActivityThreshold
            && cellFunctionConstructionUnlimitedEnergy == other.cellFunctionConstructionUnlimitedEnergy
            && cellFunctionMuscleBendingAcceleration == other.cellFunctionMuscleBendingAcceleration
            && cellFunctionMuscleBendingAccelerationThreshold == other.cellFunctionMuscleBendingAccelerationThreshold
            && cellFunctionConstructorMutationColor == other.cellFunctionConstructorMutationColor
            && cellFunctionConstructorMutationSelfReplication == other.cellFunctionConstructorMutationSelfReplication
        ;
    }

    bool operator!=(SimulationParameters const& other) const { return !operator==(other); }
};
