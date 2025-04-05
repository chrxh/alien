#include "ParametersValidationService.h"

void ParametersValidationService::validateAndCorrect(SimulationParameters& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            parameters.attackerSameMutantProtection[i][j] = std::max(0.0f, std::min(1.0f, parameters.attackerSameMutantProtection[i][j]));
            parameters.baseValues.attackerNewComplexMutantProtection[i][j] =
                std::max(0.0f, std::min(1.0f, parameters.baseValues.attackerNewComplexMutantProtection[i][j]));
        }
        parameters.radiationAbsorptionHighVelocityPenalty[i] = std::max(0.0f, parameters.radiationAbsorptionHighVelocityPenalty[i]);
        parameters.radiationAbsorptionLowConnectionPenalty[i] = std::max(0.0f, parameters.radiationAbsorptionLowConnectionPenalty[i]);
        parameters.externalEnergyConditionalInflowFactor[i] = std::max(0.0f, std::min(1.0f, parameters.externalEnergyConditionalInflowFactor[i]));
        parameters.attackerSensorDetectionFactor[i] = std::max(0.0f, std::min(1.0f, parameters.attackerSensorDetectionFactor[i]));
        parameters.externalEnergyInflowFactor[i] = std::max(0.0f, std::min(1.0f, parameters.externalEnergyInflowFactor[i]));
        parameters.minCellEnergy.baseValue[i] = std::min(parameters.minCellEnergy.baseValue[i], parameters.normalCellEnergy.value[i] * 0.95f);
        parameters.particleSplitEnergy.value[i] = std::max(0.0f, parameters.particleSplitEnergy.value[i]);
        parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty[i] =
            std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty[i]));
        parameters.baseValues.radiationAbsorptionLowVelocityPenalty[i] =
            std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorptionLowVelocityPenalty[i]));
        parameters.genomeComplexitySizeFactor[i] = std::max(0.0f, parameters.genomeComplexitySizeFactor[i]);
        parameters.genomeComplexityRamificationFactor[i] = std::max(0.0f, parameters.genomeComplexityRamificationFactor[i]);
        parameters.defenderAntiAttackerStrength.value[i] = std::max(0.0f, parameters.defenderAntiAttackerStrength.value[i]);
        parameters.defenderAntiInjectorStrength.value[i] = std::max(0.0f, parameters.defenderAntiInjectorStrength.value[i]);
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
            zone.values.attackerNewComplexMutantProtection[i][j] =
                std::max(0.0f, std::min(1.0f, zone.values.attackerNewComplexMutantProtection[i][j]));
        }
        zone.values.radiationAbsorptionLowGenomeComplexityPenalty[i] =
            std::max(0.0f, std::min(1.0f, zone.values.radiationAbsorptionLowGenomeComplexityPenalty[i]));
        zone.values.radiationAbsorptionLowVelocityPenalty[i] = std::max(0.0f, std::min(1.0f, zone.values.radiationAbsorptionLowVelocityPenalty[i]));
    }
}
