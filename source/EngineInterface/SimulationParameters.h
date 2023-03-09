#pragma once

#include <cstdint>

#include "SimulationParametersSpotValues.h"
#include "RadiationSource.h"
#include "SimulationParametersSpot.h"

struct FluidMotion
{
    float smoothingLength = 0.66f;
    float viscosityStrength = 0.1f;
    float pressureStrength = 0.1f;

    bool operator==(FluidMotion const& other) const
    {
        return smoothingLength == other.smoothingLength && viscosityStrength == other.viscosityStrength && pressureStrength == other.pressureStrength;
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

    FloatByColor cellNormalEnergy = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    float cellMinDistance = 0.3f;         
    float cellMaxForceDecayProb = 0.2f;
    int cellNumExecutionOrderNumbers = 6;

    FloatByColor radiationAbsorption = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float radiationProb = 0.03f;
    float radiationVelocityMultiplier = 1.0f;
    float radiationVelocityPerturbation = 0.5f;
    IntByColor radiationMinCellAge = {0, 0, 0, 0, 0, 0, 0};
    FloatByColor highRadiationFactor = {0, 0, 0, 0, 0, 0, 0};
    FloatByColor highRadiationMinCellEnergy = {500.0f, 500.0f, 500.0f, 500.0f, 500.0f, 500.0f, 500.0f};
    bool clusterDecay = false;
    FloatByColor clusterDecayProb = {0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0001f};
    
    bool cellFunctionConstructionUnlimitedEnergy = false;
    FloatByColor cellFunctionConstructorOffspringDistance = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    FloatByColor cellFunctionConstructorConnectingCellMaxDistance = {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f};
    FloatByColor cellFunctionConstructorActivityThreshold = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};

    bool cellFunctionConstructorMutationColor = false;
    bool cellFunctionConstructorMutationSelfReplication = false;

    FloatByColor cellFunctionInjectorRadius = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    int cellFunctionInjectorDurationColorMatrix[MAX_COLORS][MAX_COLORS] = {
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1}
    };
    float cellFunctionInjectorActivityThreshold = 0.1f;

    FloatByColor cellFunctionAttackerRadius = {1.6f, 1.6f, 1.6f, 1.6f, 1.6f, 1.6f, 1.6f};
    FloatByColor cellFunctionAttackerStrength = {0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f};
    FloatByColor cellFunctionAttackerEnergyDistributionRadius = {3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f};
    bool cellFunctionAttackerEnergyDistributionSameColor = true;
    FloatByColor cellFunctionAttackerEnergyDistributionValue = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    FloatByColor cellFunctionAttackerColorInhomogeneityFactor = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    FloatByColor cellFunctionAttackerVelocityPenalty = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float cellFunctionAttackerActivityThreshold = 0.1f;

    FloatByColor cellFunctionDefenderAgainstAttackerStrength = {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f};
    FloatByColor cellFunctionDefenderAgainstInjectorStrength = {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f};

    bool cellFunctionTransmitterEnergyDistributionSameColor = true;
    FloatByColor cellFunctionTransmitterEnergyDistributionRadius = {3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f, 3.6f};
    FloatByColor cellFunctionTransmitterEnergyDistributionValue = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};

    FloatByColor cellFunctionMuscleContractionExpansionDelta = {0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f};
    FloatByColor cellFunctionMuscleMovementAcceleration = {0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f};
    FloatByColor cellFunctionMuscleBendingAngle = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
    FloatByColor cellFunctionMuscleBendingAcceleration = {0.08f, 0.08f, 0.08f, 0.08f, 0.08f, 0.08f, 0.08f};
    float cellFunctionMuscleBendingAccelerationThreshold = 0.1f;

    FloatByColor cellFunctionSensorRange = {255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f};
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
            if (cellFunctionAttackerVelocityPenalty[i] != other.cellFunctionAttackerVelocityPenalty[i]) {
                return false;
            }
            if (cellFunctionInjectorRadius[i] != other.cellFunctionInjectorRadius[i]) {
                return false;
            }
            if (cellFunctionDefenderAgainstAttackerStrength[i] != other.cellFunctionDefenderAgainstAttackerStrength[i]) {
                return false;
            }
            if (cellFunctionDefenderAgainstInjectorStrength[i] != other.cellFunctionDefenderAgainstInjectorStrength[i]) {
                return false;
            }
            if (cellFunctionTransmitterEnergyDistributionRadius[i] != other.cellFunctionTransmitterEnergyDistributionRadius[i]) {
                return false;
            }
            if (cellFunctionTransmitterEnergyDistributionValue[i] != other.cellFunctionTransmitterEnergyDistributionValue[i]) {
                return false;
            }
            if (cellFunctionMuscleBendingAcceleration[i] != other.cellFunctionMuscleBendingAcceleration[i]) {
                return false;
            }
            if (cellFunctionMuscleContractionExpansionDelta[i] != other.cellFunctionMuscleContractionExpansionDelta[i]) {
                return false;
            }
            if (cellFunctionMuscleMovementAcceleration[i] != other.cellFunctionMuscleMovementAcceleration[i]) {
                return false;
            }
            if (cellFunctionMuscleBendingAngle[i] != other.cellFunctionMuscleBendingAngle[i]) {
                return false;
            }
            if (cellFunctionMuscleContractionExpansionDelta[i] != other.cellFunctionMuscleContractionExpansionDelta[i]) {
                return false;
            }
            if (cellFunctionMuscleMovementAcceleration[i] != other.cellFunctionMuscleMovementAcceleration[i]) {
                return false;
            }
            if (cellFunctionMuscleBendingAngle[i] != other.cellFunctionMuscleBendingAngle[i]) {
                return false;
            }
            if (cellFunctionAttackerRadius[i] != other.cellFunctionAttackerRadius[i]) {
                return false;
            }
            if (cellFunctionAttackerEnergyDistributionRadius[i] != other.cellFunctionAttackerEnergyDistributionRadius[i]) {
                return false;
            }
            if (cellFunctionAttackerEnergyDistributionValue[i] != other.cellFunctionAttackerEnergyDistributionValue[i]) {
                return false;
            }
            if (cellFunctionAttackerStrength[i] != other.cellFunctionAttackerStrength[i]) {
                return false;
            }
            if (cellFunctionSensorRange[i] != other.cellFunctionSensorRange[i]) {
                return false;
            }
            if (cellFunctionAttackerColorInhomogeneityFactor[i] != other.cellFunctionAttackerColorInhomogeneityFactor[i]) {
                return false;
            }
            if (cellFunctionConstructorActivityThreshold[i] != other.cellFunctionConstructorActivityThreshold[i]) {
                return false;
            }
            if (cellFunctionConstructorConnectingCellMaxDistance[i] != other.cellFunctionConstructorConnectingCellMaxDistance[i]) {
                return false;
            }
            if (cellFunctionConstructorOffspringDistance[i] != other.cellFunctionConstructorOffspringDistance[i]) {
                return false;
            }
            if (clusterDecayProb[i] != other.clusterDecayProb[i]) {
                return false;
            }
            if (cellNormalEnergy[i] != other.cellNormalEnergy[i]) {
                return false;
            }
            if (radiationAbsorption[i] != other.radiationAbsorption[i]) {
                return false;
            }
            if (highRadiationFactor[i] != other.highRadiationFactor[i]) {
                return false;
            }
            if (highRadiationMinCellEnergy[i] != other.highRadiationMinCellEnergy[i]) {
                return false;
            }
            if (radiationMinCellAge[i] != other.radiationMinCellAge[i]) {
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
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                if (cellFunctionInjectorDurationColorMatrix[i][j] != other.cellFunctionInjectorDurationColorMatrix[i][j]) {
                    return false;
                }
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
            && cellNumExecutionOrderNumbers == other.cellNumExecutionOrderNumbers
            && radiationProb == other.radiationProb && radiationVelocityMultiplier == other.radiationVelocityMultiplier
            && radiationVelocityPerturbation == other.radiationVelocityPerturbation
            && cellFunctionAttackerActivityThreshold == other.cellFunctionAttackerActivityThreshold
            && particleTransformationMaxGenomeSize == other.particleTransformationMaxGenomeSize
            && cellFunctionAttackerEnergyDistributionSameColor == other.cellFunctionAttackerEnergyDistributionSameColor
            && cellFunctionTransmitterEnergyDistributionSameColor == other.cellFunctionTransmitterEnergyDistributionSameColor
            && particleTransformationAllowed == other.particleTransformationAllowed
            && particleTransformationRandomCellFunction == other.particleTransformationRandomCellFunction
            && particleTransformationMaxGenomeSize == other.particleTransformationMaxGenomeSize
            && clusterDecay == other.clusterDecay
            && cellFunctionSensorActivityThreshold == other.cellFunctionSensorActivityThreshold
            && cellFunctionConstructionUnlimitedEnergy == other.cellFunctionConstructionUnlimitedEnergy
            && cellFunctionMuscleBendingAccelerationThreshold == other.cellFunctionMuscleBendingAccelerationThreshold
            && cellFunctionConstructorMutationColor == other.cellFunctionConstructorMutationColor
            && cellFunctionConstructorMutationSelfReplication == other.cellFunctionConstructorMutationSelfReplication
        ;
    }

    bool operator!=(SimulationParameters const& other) const { return !operator==(other); }
};
