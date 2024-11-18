#include "SimulationParameters.h"

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

    for (int i = 0, j = sizeof(Char64) / sizeof(char); i < j; ++i) {
        if (projectName[i] != other.projectName[i]) {
            return false;
        }
        if (projectName[i] == '\0') {
            break;
        }
    }

    for (int i = 0; i < MAX_COLORS; ++i) {
        if (cellMaxBindingDistance[i] != other.cellMaxBindingDistance[i]) {
            return false;
        }
        if (externalEnergyBackflowFactor[i] != other.externalEnergyBackflowFactor[i]) {
            return false;
        }
        if (radiationAbsorptionHighVelocityPenalty[i] != other.radiationAbsorptionHighVelocityPenalty[i]) {
            return false;
        }
        if (particleSplitEnergy[i] != other.particleSplitEnergy[i]) {
            return false;
        }
        if (cellFunctionAttackerSensorDetectionFactor[i] != other.cellFunctionAttackerSensorDetectionFactor[i]) {
            return false;
        }
        if (externalEnergyConditionalInflowFactor[i] != other.externalEnergyConditionalInflowFactor[i]) {
            return false;
        }
        if (cellMaxAge[i] != other.cellMaxAge[i]) {
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
        if (cellFunctionConstructorSignalThreshold[i] != other.cellFunctionConstructorSignalThreshold[i]) {
            return false;
        }
        if (cellFunctionConstructorConnectingCellMaxDistance[i] != other.cellFunctionConstructorConnectingCellMaxDistance[i]) {
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
        if (externalEnergyInflowFactor[i] != other.externalEnergyInflowFactor[i]) {
            return false;
        }
        if (genomeComplexityRamificationFactor[i] != other.genomeComplexityRamificationFactor[i]) {
            return false;
        }
        if (genomeComplexitySizeFactor[i] != other.genomeComplexitySizeFactor[i]) {
            return false;
        }
        if (cellEmergentMaxAge[i] != other.cellEmergentMaxAge[i]) {
            return false;
        }
        if (genomeComplexityNeuronFactor[i] != other.genomeComplexityNeuronFactor[i]) {
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
            if (cellFunctionAttackerSameMutantPenalty[i][j] != other.cellFunctionAttackerSameMutantPenalty[i][j]) {
                return false;
            }
        }
    }
    if (numRadiationSources != other.numRadiationSources) {
        return false;
    }
    for (int i = 0; i < numRadiationSources; ++i) {
        if (radiationSources[i] != other.radiationSources[i]) {
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

    return backgroundColor == other.backgroundColor && cellColoring == other.cellColoring && cellGlowColoring == other.cellGlowColoring
        && zoomLevelNeuronalActivity == other.zoomLevelNeuronalActivity && baseValues == other.baseValues && timestepSize == other.timestepSize
        && cellMaxVelocity == other.cellMaxVelocity && cellMinDistance == other.cellMinDistance && cellMaxForceDecayProb == other.cellMaxForceDecayProb
        && cellNumExecutionOrderNumbers == other.cellNumExecutionOrderNumbers && radiationProb == other.radiationProb
        && radiationVelocityMultiplier == other.radiationVelocityMultiplier && radiationVelocityPerturbation == other.radiationVelocityPerturbation
        && cellFunctionAttackerSignalThreshold == other.cellFunctionAttackerSignalThreshold
        && particleTransformationMaxGenomeSize == other.particleTransformationMaxGenomeSize
        && cellFunctionTransmitterEnergyDistributionSameCreature == other.cellFunctionTransmitterEnergyDistributionSameCreature
        && particleTransformationAllowed == other.particleTransformationAllowed
        && particleTransformationRandomCellFunction == other.particleTransformationRandomCellFunction
        && particleTransformationMaxGenomeSize == other.particleTransformationMaxGenomeSize && cellDeathConsequences == other.cellDeathConsequences
        && cellFunctionSensorSignalThreshold == other.cellFunctionSensorSignalThreshold
        && cellFunctionMuscleBendingAccelerationThreshold == other.cellFunctionMuscleBendingAccelerationThreshold
        && cellFunctionConstructorMutationSelfReplication == other.cellFunctionConstructorMutationSelfReplication
        && cellMaxAgeBalancer == other.cellMaxAgeBalancer && cellMaxAgeBalancerInterval == other.cellMaxAgeBalancerInterval
        && cellFunctionConstructorMutationPreventDepthIncrease == other.cellFunctionConstructorMutationPreventDepthIncrease
        && cellFunctionConstructorCheckCompletenessForSelfReplication == other.cellFunctionConstructorCheckCompletenessForSelfReplication
        && cellFunctionAttackerDestroyCells == other.cellFunctionAttackerDestroyCells
        && cellFunctionReconnectorSignalThreshold == other.cellFunctionReconnectorSignalThreshold
        && cellFunctionDetonatorSignalThreshold == other.cellFunctionDetonatorSignalThreshold && features == other.features
        && highlightedCellFunction == other.highlightedCellFunction && borderlessRendering == other.borderlessRendering
        && markReferenceDomain == other.markReferenceDomain && gridLines == other.gridLines && externalEnergy == other.externalEnergy
        && attackVisualization == other.attackVisualization && cellInactiveMaxAgeActivated == other.cellInactiveMaxAgeActivated
        && cellGlowStrength == other.cellGlowStrength && cellGlowRadius == other.cellGlowRadius
        && cellResetAgeAfterActivation == other.cellResetAgeAfterActivation && cellRadius == other.cellRadius
        && cellEmergentMaxAgeActivated == other.cellEmergentMaxAgeActivated
        && cellFunctionMuscleMovementTowardTargetedObject == other.cellFunctionMuscleMovementTowardTargetedObject
        && legacyCellFunctionMuscleMovementAngleFromSensor == other.legacyCellFunctionMuscleMovementAngleFromSensor
        && muscleMovementVisualization == other.muscleMovementVisualization
        && externalEnergyInflowOnlyForNonSelfReplicators == other.externalEnergyInflowOnlyForNonSelfReplicators
        && externalEnergyBackflowLimit == other.externalEnergyBackflowLimit && baseStrengthRatioPinned == other.baseStrengthRatioPinned;
}