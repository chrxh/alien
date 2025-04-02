#include "ParametersValidationService.h"

void ParametersValidationService::validateAndCorrect(SimulationParameters& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            parameters.baseValues.attackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, parameters.baseValues.attackerFoodChainColorMatrix[i][j]));
            parameters.attackerSameMutantProtection[i][j] = std::max(0.0f, std::min(1.0f, parameters.attackerSameMutantProtection[i][j]));
            parameters.baseValues.attackerNewComplexMutantProtection[i][j] =
                std::max(0.0f, std::min(1.0f, parameters.baseValues.attackerNewComplexMutantProtection[i][j]));
            parameters.baseValues.attackerComplexCreatureProtection[i][j] =
                std::max(0.0f, parameters.baseValues.attackerComplexCreatureProtection[i][j]);
        }
        parameters.baseValues.radiationAbsorption[i] = std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorption[i]));
        parameters.radiationAbsorptionHighVelocityPenalty[i] = std::max(0.0f, parameters.radiationAbsorptionHighVelocityPenalty[i]);
        parameters.radiationAbsorptionLowConnectionPenalty[i] = std::max(0.0f, parameters.radiationAbsorptionLowConnectionPenalty[i]);
        parameters.externalEnergyConditionalInflowFactor[i] = std::max(0.0f, std::min(1.0f, parameters.externalEnergyConditionalInflowFactor[i]));
        parameters.attackerSensorDetectionFactor[i] = std::max(0.0f, std::min(1.0f, parameters.attackerSensorDetectionFactor[i]));
        parameters.detonatorChainExplosionProbability[i] =
            std::max(0.0f, std::min(1.0f, parameters.detonatorChainExplosionProbability[i]));
        parameters.externalEnergyInflowFactor[i] = std::max(0.0f, std::min(1.0f, parameters.externalEnergyInflowFactor[i]));
        parameters.baseValues.minCellEnergy[i] = std::min(parameters.baseValues.minCellEnergy[i], parameters.normalCellEnergy[i] * 0.95f);
        parameters.particleSplitEnergy[i] = std::max(0.0f, parameters.particleSplitEnergy[i]);
        parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty[i] =
            std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty[i]));
        parameters.baseValues.radiationAbsorptionLowVelocityPenalty[i] =
            std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorptionLowVelocityPenalty[i]));
        parameters.genomeComplexitySizeFactor[i] = std::max(0.0f, parameters.genomeComplexitySizeFactor[i]);
        parameters.genomeComplexityRamificationFactor[i] = std::max(0.0f, parameters.genomeComplexityRamificationFactor[i]);
        parameters.muscleEnergyCost[i] = std::max(0.0f, std::min(5.0f, parameters.muscleEnergyCost[i]));
        parameters.defenderAntiAttackerStrength[i] = std::max(0.0f, parameters.defenderAntiAttackerStrength[i]);
        parameters.defenderAntiInjectorStrength[i] = std::max(0.0f, parameters.defenderAntiInjectorStrength[i]);
    }
    parameters.externalEnergy = std::max(0.0f, parameters.externalEnergy);
    parameters.timestepSize.value = std::max(0.01f, parameters.timestepSize.value);
    parameters.maxCellAgeBalancerInterval = std::max(1000, std::min(1000000, parameters.maxCellAgeBalancerInterval));
    parameters.cellGlowRadius = std::max(1.0f, std::min(8.0f, parameters.cellGlowRadius));

    parameters.cellCopyMutationNeuronDataWeight = std::max(0.0f, std::min(1.0f, parameters.cellCopyMutationNeuronDataWeight));
    parameters.cellCopyMutationNeuronDataBias = std::max(0.0f, std::min(1.0f, parameters.cellCopyMutationNeuronDataBias));
    parameters.cellCopyMutationNeuronDataActivationFunction = std::max(0.0f, std::min(1.0f, parameters.cellCopyMutationNeuronDataActivationFunction));
    parameters.cellCopyMutationNeuronDataReinforcement = std::max(1.0f, std::min(2.0f, parameters.cellCopyMutationNeuronDataReinforcement));
    parameters.cellCopyMutationNeuronDataDamping = std::max(1.0f, std::min(2.0f, parameters.cellCopyMutationNeuronDataDamping));
    parameters.cellCopyMutationNeuronDataOffset = std::max(0.0f, std::min(1.0f, parameters.cellCopyMutationNeuronDataOffset));

    for (int i = 0; i < parameters.numZones.value; ++i) {
        validateAndCorrect(parameters.zone[i], parameters);
    }
}

void ParametersValidationService::validateAndCorrect(RadiationSource& source) const
{
    if (source.shape.type == RadiationSourceShapeType_Circular) {
        source.shape.alternatives.circularRadiationSource.radius = std::max(1.0f, source.shape.alternatives.circularRadiationSource.radius);
    }
    if (source.shape.type == RadiationSourceShapeType_Rectangular) {
        source.shape.alternatives.rectangularRadiationSource.width = std::max(1.0f, source.shape.alternatives.rectangularRadiationSource.width);
        source.shape.alternatives.rectangularRadiationSource.height = std::max(1.0f, source.shape.alternatives.rectangularRadiationSource.height);
    }
}

void ParametersValidationService::validateAndCorrect(SimulationParametersZone& zone, SimulationParameters const& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            zone.values.attackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, zone.values.attackerFoodChainColorMatrix[i][j]));
            zone.values.attackerComplexCreatureProtection[i][j] = std::max(0.0f, zone.values.attackerComplexCreatureProtection[i][j]);
            zone.values.attackerNewComplexMutantProtection[i][j] =
                std::max(0.0f, std::min(1.0f, zone.values.attackerNewComplexMutantProtection[i][j]));
        }
        zone.values.radiationAbsorption[i] = std::max(0.0f, std::min(1.0f, zone.values.radiationAbsorption[i]));
        zone.values.minCellEnergy[i] = std::min(parameters.baseValues.minCellEnergy[i], parameters.normalCellEnergy[i] * 0.95f);
        zone.values.radiationAbsorptionLowGenomeComplexityPenalty[i] =
            std::max(0.0f, std::min(1.0f, zone.values.radiationAbsorptionLowGenomeComplexityPenalty[i]));
        zone.values.radiationAbsorptionLowVelocityPenalty[i] = std::max(0.0f, std::min(1.0f, zone.values.radiationAbsorptionLowVelocityPenalty[i]));
    }
}
