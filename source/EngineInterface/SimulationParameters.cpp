#include "SimulationParameters.h"

bool Features::operator==(Features const& other) const
{
    return externalEnergy == other.externalEnergy && colorTransitions == other.colorTransitions;
}

bool SimulationParameters::operator==(SimulationParameters const& other) const
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
        if (particleSplitEnergy[i] != other.particleSplitEnergy[i]) {
            return false;
        }
        if (cellFunctionAttackerSensorDetectionFactor[i] != other.cellFunctionAttackerSensorDetectionFactor[i]) {
            return false;
        }
        if (cellFunctionConstructorPumpEnergyFactor[i] != other.cellFunctionConstructorPumpEnergyFactor[i]) {
            return false;
        }
        if (cellMaxAge[i] != other.cellMaxAge[i]) {
            return false;
        }
        if (radiationAbsorptionVelocityPenalty[i] != other.radiationAbsorptionVelocityPenalty[i]) {
            return false;
        }
        if (radiationAbsorptionLowConnectionPenalty[i] != other.radiationAbsorptionLowConnectionPenalty[i]) {
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
        if (highRadiationFactor[i] != other.highRadiationFactor[i]) {
            return false;
        }
        if (highRadiationMinCellEnergy[i] != other.highRadiationMinCellEnergy[i]) {
            return false;
        }
        if (radiationMinCellAge[i] != other.radiationMinCellAge[i]) {
            return false;
        }
        if (cellFunctionReconnectorRadius[i] != other.cellFunctionReconnectorRadius[i]) {
            return false;
        }
        if (cellFunctionDetonatorRadius[i] != other.cellFunctionDetonatorRadius[i]) {
            return false;
        }
        if (cellFunctionDetonatorChainExplosionProbability[i] != other.cellFunctionDetonatorChainExplosionProbability[i]) {
            return false;
        }
        if (cellFunctionConstructorExternalEnergy[i] != other.cellFunctionConstructorExternalEnergy[i]) {
            return false;
        }
        if (cellFunctionConstructorExternalEnergySupplyRate[i] != other.cellFunctionConstructorExternalEnergySupplyRate[i]) {
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
            if (cellFunctionConstructorMutationColorTransitions[i][j] != other.cellFunctionConstructorMutationColorTransitions[i][j]) {
                return false;
            }
            if (cellFunctionInjectorDurationColorMatrix[i][j] != other.cellFunctionInjectorDurationColorMatrix[i][j]) {
                return false;
            }
            if (cellFunctionAttackerGenomeSizeBonus[i][j] != other.cellFunctionAttackerGenomeSizeBonus[i][j]) {
                return false;
            }
            if (cellFunctionAttackerSameMutantPenalty[i][j] != other.cellFunctionAttackerSameMutantPenalty[i][j]) {
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

    return backgroundColor == other.backgroundColor && cellColorization == other.cellColorization
        && zoomLevelNeuronalActivity == other.zoomLevelNeuronalActivity && showDetonations == other.showDetonations && baseValues == other.baseValues
        && timestepSize == other.timestepSize && cellMaxVelocity == other.cellMaxVelocity && cellMaxBindingDistance == other.cellMaxBindingDistance
        && cellMinDistance == other.cellMinDistance && cellMaxForceDecayProb == other.cellMaxForceDecayProb
        && cellNumExecutionOrderNumbers == other.cellNumExecutionOrderNumbers && radiationProb == other.radiationProb
        && radiationVelocityMultiplier == other.radiationVelocityMultiplier && radiationVelocityPerturbation == other.radiationVelocityPerturbation
        && cellFunctionAttackerActivityThreshold == other.cellFunctionAttackerActivityThreshold
        && particleTransformationMaxGenomeSize == other.particleTransformationMaxGenomeSize
        && cellFunctionTransmitterEnergyDistributionSameCreature == other.cellFunctionTransmitterEnergyDistributionSameCreature
        && particleTransformationAllowed == other.particleTransformationAllowed
        && particleTransformationRandomCellFunction == other.particleTransformationRandomCellFunction
        && particleTransformationMaxGenomeSize == other.particleTransformationMaxGenomeSize && clusterDecay == other.clusterDecay
        && cellFunctionSensorActivityThreshold == other.cellFunctionSensorActivityThreshold
        && cellFunctionMuscleBendingAccelerationThreshold == other.cellFunctionMuscleBendingAccelerationThreshold
        && cellFunctionConstructorMutationSelfReplication == other.cellFunctionConstructorMutationSelfReplication
        && cellMaxAgeBalancer == other.cellMaxAgeBalancer && cellMaxAgeBalancerInterval == other.cellMaxAgeBalancerInterval
        && cellFunctionConstructorMutationPreventDepthIncrease == other.cellFunctionConstructorMutationPreventDepthIncrease
        && cellFunctionConstructorCheckCompletenessForSelfReplication == other.cellFunctionConstructorCheckCompletenessForSelfReplication
        && cellFunctionAttackerDestroyCells == other.cellFunctionAttackerDestroyCells
        && cellFunctionReconnectorActivityThreshold == other.cellFunctionReconnectorActivityThreshold
        && cellFunctionDetonatorActivityThreshold == other.cellFunctionDetonatorActivityThreshold && features == other.features;
}