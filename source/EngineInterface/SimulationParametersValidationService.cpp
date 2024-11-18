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

    for (int i = 0; i < parameters.numSpots; ++i) {
        validateAndCorrect(parameters.spots[i], parameters);
    }
}

void SimulationParametersValidationService::validateAndCorrect(SimulationParametersSpot& spot, SimulationParameters const& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j] =
                std::max(0.0f, std::min(1.0f, spot.values.cellFunctionAttackerFoodChainColorMatrix[i][j]));
            spot.values.cellFunctionAttackerGenomeComplexityBonus[i][j] = std::max(0.0f, spot.values.cellFunctionAttackerGenomeComplexityBonus[i][j]);
            spot.values.cellFunctionAttackerNewComplexMutantPenalty[i][j] =
                std::max(0.0f, std::min(1.0f, spot.values.cellFunctionAttackerNewComplexMutantPenalty[i][j]));
        }
        spot.values.radiationAbsorption[i] = std::max(0.0f, std::min(1.0f, spot.values.radiationAbsorption[i]));
        spot.values.cellMinEnergy[i] = std::min(parameters.baseValues.cellMinEnergy[i], parameters.cellNormalEnergy[i] * 0.95f);
        spot.values.radiationAbsorptionLowGenomeComplexityPenalty[i] =
            std::max(0.0f, std::min(1.0f, spot.values.radiationAbsorptionLowGenomeComplexityPenalty[i]));
        spot.values.radiationAbsorptionLowVelocityPenalty[i] = std::max(0.0f, std::min(1.0f, spot.values.radiationAbsorptionLowVelocityPenalty[i]));
    }
    spot.values.cellMaxBindingEnergy = std::max(10.0f, spot.values.cellMaxBindingEnergy);
}
