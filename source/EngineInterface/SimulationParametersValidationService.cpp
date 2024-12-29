#include "SimulationParametersValidationService.h"

void SimulationParametersValidationService::validateAndCorrect(SimulationParameters& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, parameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[i][j]));
            parameters.cellFunctionAttackerSameMutantPenalty[i][j] = std::max(0.0f, std::min(1.0f, parameters.cellFunctionAttackerSameMutantPenalty[i][j]));
            parameters.baseValues.cellFunctionAttackerNewComplexMutantPenalty[i][j] =
                std::max(0.0f, std::min(1.0f, parameters.baseValues.cellFunctionAttackerNewComplexMutantPenalty[i][j]));
            parameters.baseValues.cellFunctionAttackerGenomeComplexityBonus[i][j] =
                std::max(0.0f, parameters.baseValues.cellFunctionAttackerGenomeComplexityBonus[i][j]);
        }
        parameters.baseValues.radiationAbsorption[i] = std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorption[i]));
        parameters.radiationAbsorptionHighVelocityPenalty[i] = std::max(0.0f, parameters.radiationAbsorptionHighVelocityPenalty[i]);
        parameters.radiationAbsorptionLowConnectionPenalty[i] = std::max(0.0f, parameters.radiationAbsorptionLowConnectionPenalty[i]);
        parameters.externalEnergyConditionalInflowFactor[i] = std::max(0.0f, std::min(1.0f, parameters.externalEnergyConditionalInflowFactor[i]));
        parameters.cellFunctionAttackerSensorDetectionFactor[i] = std::max(0.0f, std::min(1.0f, parameters.cellFunctionAttackerSensorDetectionFactor[i]));
        parameters.cellFunctionDetonatorChainExplosionProbability[i] =
            std::max(0.0f, std::min(1.0f, parameters.cellFunctionDetonatorChainExplosionProbability[i]));
        parameters.externalEnergyInflowFactor[i] = std::max(0.0f, std::min(1.0f, parameters.externalEnergyInflowFactor[i]));
        parameters.baseValues.cellMinEnergy[i] = std::min(parameters.baseValues.cellMinEnergy[i], parameters.cellNormalEnergy[i] * 0.95f);
        parameters.particleSplitEnergy[i] = std::max(0.0f, parameters.particleSplitEnergy[i]);
        parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty[i] =
            std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorptionLowGenomeComplexityPenalty[i]));
        parameters.baseValues.radiationAbsorptionLowVelocityPenalty[i] =
            std::max(0.0f, std::min(1.0f, parameters.baseValues.radiationAbsorptionLowVelocityPenalty[i]));
        parameters.genomeComplexitySizeFactor[i] = std::max(0.0f, parameters.genomeComplexitySizeFactor[i]);
        parameters.genomeComplexityRamificationFactor[i] = std::max(0.0f, parameters.genomeComplexityRamificationFactor[i]);
        parameters.genomeComplexityNeuronFactor[i] = std::max(0.0f, parameters.genomeComplexityNeuronFactor[i]);
    }
    parameters.externalEnergy = std::max(0.0f, parameters.externalEnergy);
    parameters.baseValues.cellMaxBindingEnergy = std::max(10.0f, parameters.baseValues.cellMaxBindingEnergy);
    parameters.timestepSize = std::max(0.0f, parameters.timestepSize);
    parameters.cellMaxAgeBalancerInterval = std::max(1000, std::min(1000000, parameters.cellMaxAgeBalancerInterval));
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
    if (source.shapeType == RadiationSourceShapeType_Circular) {
        source.shapeData.circularRadiationSource.radius = std::max(1.0f, source.shapeData.circularRadiationSource.radius);
    }
    if (source.shapeType == RadiationSourceShapeType_Rectangular) {
        source.shapeData.rectangularRadiationSource.width = std::max(1.0f, source.shapeData.rectangularRadiationSource.width);
        source.shapeData.rectangularRadiationSource.height = std::max(1.0f, source.shapeData.rectangularRadiationSource.height);
    }
}

void SimulationParametersValidationService::validateAndCorrect(SimulationParametersZone& zone, SimulationParameters const& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            zone.values.cellFunctionAttackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, zone.values.cellFunctionAttackerFoodChainColorMatrix[i][j]));
            zone.values.cellFunctionAttackerGenomeComplexityBonus[i][j] = std::max(0.0f, zone.values.cellFunctionAttackerGenomeComplexityBonus[i][j]);
            zone.values.cellFunctionAttackerNewComplexMutantPenalty[i][j] =
                std::max(0.0f, std::min(1.0f, zone.values.cellFunctionAttackerNewComplexMutantPenalty[i][j]));
        }
        zone.values.radiationAbsorption[i] = std::max(0.0f, std::min(1.0f, zone.values.radiationAbsorption[i]));
        zone.values.cellMinEnergy[i] = std::min(parameters.baseValues.cellMinEnergy[i], parameters.cellNormalEnergy[i] * 0.95f);
        zone.values.radiationAbsorptionLowGenomeComplexityPenalty[i] =
            std::max(0.0f, std::min(1.0f, zone.values.radiationAbsorptionLowGenomeComplexityPenalty[i]));
        zone.values.radiationAbsorptionLowVelocityPenalty[i] = std::max(0.0f, std::min(1.0f, zone.values.radiationAbsorptionLowVelocityPenalty[i]));
    }
    zone.values.cellMaxBindingEnergy = std::max(10.0f, zone.values.cellMaxBindingEnergy);
}
