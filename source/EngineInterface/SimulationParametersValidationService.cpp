#include "SimulationParametersValidationService.h"

void SimulationParametersValidationService::validateAndCorrect(SimulationParameters& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            parameters.baseValues.cellTypeAttackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, parameters.baseValues.cellTypeAttackerFoodChainColorMatrix[i][j]));
            parameters.attackerSameMutantPenalty[i][j] = std::max(0.0f, std::min(1.0f, parameters.attackerSameMutantPenalty[i][j]));
            parameters.baseValues.cellTypeAttackerNewComplexMutantPenalty[i][j] =
                std::max(0.0f, std::min(1.0f, parameters.baseValues.cellTypeAttackerNewComplexMutantPenalty[i][j]));
            parameters.baseValues.cellTypeAttackerGenomeComplexityBonus[i][j] =
                std::max(0.0f, parameters.baseValues.cellTypeAttackerGenomeComplexityBonus[i][j]);
        }
        parameters.baseValues.radiationAbsorption[i] = std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorption[i]));
        parameters.radiationAbsorptionHighVelocityPenalty[i] = std::max(0.0f, parameters.radiationAbsorptionHighVelocityPenalty[i]);
        parameters.radiationAbsorptionLowConnectionPenalty[i] = std::max(0.0f, parameters.radiationAbsorptionLowConnectionPenalty[i]);
        parameters.externalEnergyConditionalInflowFactor[i] = std::max(0.0f, std::min(1.0f, parameters.externalEnergyConditionalInflowFactor[i]));
        parameters.attackerSensorDetectionFactor[i] = std::max(0.0f, std::min(1.0f, parameters.attackerSensorDetectionFactor[i]));
        parameters.detonatorChainExplosionProbability[i] =
            std::max(0.0f, std::min(1.0f, parameters.detonatorChainExplosionProbability[i]));
        parameters.externalEnergyInflowFactor[i] = std::max(0.0f, std::min(1.0f, parameters.externalEnergyInflowFactor[i]));
        parameters.baseValues.cellMinEnergy[i] = std::min(parameters.baseValues.cellMinEnergy[i], parameters.normalCellEnergy[i] * 0.95f);
        parameters.particleSplitEnergy[i] = std::max(0.0f, parameters.particleSplitEnergy[i]);
        parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty[i] =
            std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty[i]));
        parameters.baseValues.radiationAbsorptionLowVelocityPenalty[i] =
            std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorptionLowVelocityPenalty[i]));
        parameters.genomeComplexitySizeFactor[i] = std::max(0.0f, parameters.genomeComplexitySizeFactor[i]);
        parameters.genomeComplexityRamificationFactor[i] = std::max(0.0f, parameters.genomeComplexityRamificationFactor[i]);
        parameters.genomeComplexityNeuronFactor[i] = std::max(0.0f, parameters.genomeComplexityNeuronFactor[i]);
        parameters.muscleEnergyCost[i] = std::max(0.0f, std::min(5.0f, parameters.muscleEnergyCost[i]));
        parameters.defenderAntiAttackerStrength[i] = std::max(0.0f, parameters.defenderAntiAttackerStrength[i]);
        parameters.defenderAntiInjectorStrength[i] = std::max(0.0f, parameters.defenderAntiInjectorStrength[i]);
    }
    parameters.externalEnergy = std::max(0.0f, parameters.externalEnergy);
    parameters.baseValues.cellMaxBindingEnergy = std::max(10.0f, parameters.baseValues.cellMaxBindingEnergy);
    parameters.timestepSize = std::max(0.0f, parameters.timestepSize);
    parameters.maxCellAgeBalancerInterval = std::max(1000, std::min(1000000, parameters.maxCellAgeBalancerInterval));
    parameters.cellGlowRadius = std::max(1.0f, std::min(8.0f, parameters.cellGlowRadius));
    parameters.cellGlowStrength = std::max(0.0f, std::min(1.0f, parameters.cellGlowStrength));

    parameters.cellCopyMutationNeuronDataWeight = std::max(0.0f, std::min(1.0f, parameters.cellCopyMutationNeuronDataWeight));
    parameters.cellCopyMutationNeuronDataBias = std::max(0.0f, std::min(1.0f, parameters.cellCopyMutationNeuronDataBias));
    parameters.cellCopyMutationNeuronDataActivationFunction = std::max(0.0f, std::min(1.0f, parameters.cellCopyMutationNeuronDataActivationFunction));
    parameters.cellCopyMutationNeuronDataReinforcement = std::max(1.0f, std::min(2.0f, parameters.cellCopyMutationNeuronDataReinforcement));
    parameters.cellCopyMutationNeuronDataDamping = std::max(1.0f, std::min(2.0f, parameters.cellCopyMutationNeuronDataDamping));
    parameters.cellCopyMutationNeuronDataOffset = std::max(0.0f, std::min(1.0f, parameters.cellCopyMutationNeuronDataOffset));

    for (int i = 0; i < parameters.numZones; ++i) {
        validateAndCorrect(parameters.zone[i], parameters);
    }
}

void SimulationParametersValidationService::validateAndCorrect(RadiationSource& source) const
{
    if (source.shape.type == RadiationSourceShapeType_Circular) {
        source.shape.alternatives.circularRadiationSource.radius = std::max(1.0f, source.shape.alternatives.circularRadiationSource.radius);
    }
    if (source.shape.type == RadiationSourceShapeType_Rectangular) {
        source.shape.alternatives.rectangularRadiationSource.width = std::max(1.0f, source.shape.alternatives.rectangularRadiationSource.width);
        source.shape.alternatives.rectangularRadiationSource.height = std::max(1.0f, source.shape.alternatives.rectangularRadiationSource.height);
    }
}

void SimulationParametersValidationService::validateAndCorrect(SimulationParametersZone& zone, SimulationParameters const& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            zone.values.cellTypeAttackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, zone.values.cellTypeAttackerFoodChainColorMatrix[i][j]));
            zone.values.cellTypeAttackerGenomeComplexityBonus[i][j] = std::max(0.0f, zone.values.cellTypeAttackerGenomeComplexityBonus[i][j]);
            zone.values.cellTypeAttackerNewComplexMutantPenalty[i][j] =
                std::max(0.0f, std::min(1.0f, zone.values.cellTypeAttackerNewComplexMutantPenalty[i][j]));
        }
        zone.values.radiationAbsorption[i] = std::max(0.0f, std::min(1.0f, zone.values.radiationAbsorption[i]));
        zone.values.cellMinEnergy[i] = std::min(parameters.baseValues.cellMinEnergy[i], parameters.normalCellEnergy[i] * 0.95f);
        zone.values.radiationAbsorptionLowGenomeComplexityPenalty[i] =
            std::max(0.0f, std::min(1.0f, zone.values.radiationAbsorptionLowGenomeComplexityPenalty[i]));
        zone.values.radiationAbsorptionLowVelocityPenalty[i] = std::max(0.0f, std::min(1.0f, zone.values.radiationAbsorptionLowVelocityPenalty[i]));
    }
    zone.values.cellMaxBindingEnergy = std::max(10.0f, zone.values.cellMaxBindingEnergy);
}
