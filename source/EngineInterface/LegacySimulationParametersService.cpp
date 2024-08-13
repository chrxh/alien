#include "LegacySimulationParametersService.h"

#include <set>

namespace
{
    template <typename T>
    bool equals(SimulationParameters const& parameters, ColorVector<T> SimulationParametersSpotValues::*parameter, T const& value)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            if ((parameters.baseValues.*parameter)[i] != value) {
                return false;
            }
            for (int j = 0; j < parameters.numSpots; ++j) {
                if ((parameters.spots[j].values.*parameter)[i] != value) {
                    return false;
                }
            }
        }
        return true;
    }
    template <typename T>
    bool contains(SimulationParameters const& parameters, ColorVector<T> SimulationParametersSpotValues::*parameter, std::set<T> const& values)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (!values.contains((parameters.baseValues.*parameter)[i])) {
                return false;
            }
            for (int j = 0; j < parameters.numSpots; ++j) {
                if (!values.contains((parameters.spots[j].values.*parameter)[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename T>
    bool equals(ColorVector<T> const& parameter, T const& value)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (parameter[i] != value) {
                return false;
            }
        }
        return true;
    }

    template <typename T>
    bool equals(ColorMatrix<T> const& parameter, T const& value)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++i) {
                if (parameter[i][j] != value) {
                    return false;
                }
            }
        }
        return true;
    }
}

void LegacySimulationParametersService::activateFeaturesForLegacyFiles(Features const& missingFeatures, SimulationParameters& parameters)
{
    if (missingFeatures.advancedAbsorptionControl) {
        if (!equals(parameters.radiationAbsorptionHighVelocityPenalty, 0.0f) || !equals(parameters.radiationAbsorptionLowConnectionPenalty, 0.0f)) {
            parameters.features.advancedAbsorptionControl = true;
        }
    }

    if (missingFeatures.advancedAttackerControl) {
        auto advancedAttackerControlForSpot = false;
        for (int i = 0; i < parameters.numSpots; ++i) {
            auto const& spotValues = parameters.spots[i].values;
            if (!equals(spotValues.cellFunctionAttackerGeometryDeviationExponent, 0.0f)
                || !equals(spotValues.cellFunctionAttackerConnectionsMismatchPenalty, 0.0f)) {
                advancedAttackerControlForSpot = true;
            }
        }
        if (advancedAttackerControlForSpot || !equals(parameters.baseValues.cellFunctionAttackerGenomeComplexityBonus, 0.0f)
            || !equals(parameters.cellFunctionAttackerSameMutantPenalty, 0.0f)
            || !equals(parameters.cellFunctionAttackerSensorDetectionFactor, 0.0f)
            || !equals(parameters.baseValues.cellFunctionAttackerGeometryDeviationExponent, 0.0f)
            || !equals(parameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty, 0.0f)
            || !equals(parameters.cellFunctionAttackerColorInhomogeneityFactor, 1.0f)
            || !equals(parameters.cellFunctionAttackerEnergyDistributionRadius, 3.6f)
            || !equals(parameters.cellFunctionAttackerEnergyDistributionValue, 10.0f)) {
            parameters.features.advancedAttackerControl = true;
        }
    }

    if (missingFeatures.externalEnergyControl) {
        if (parameters.externalEnergy != 0.0f || !equals(parameters.externalEnergyInflowFactor, 0.0f)
            || !equals(parameters.externalEnergyConditionalInflowFactor, 0.0f) || !equals(parameters.externalEnergyBackflowFactor, 0.0f)) {
            parameters.features.externalEnergyControl = true;
        }
    }

    if (missingFeatures.cellColorTransitionRules) {
        if (!contains(parameters, &SimulationParametersSpotValues::cellColorTransitionDuration, {0, Infinity<int>::value})) {
            parameters.features.cellColorTransitionRules = true;
        }
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (parameters.baseValues.cellColorTransitionTargetColor[i] != i) {
                parameters.features.cellColorTransitionRules = true;
                break;
            }
            for (int j = 0; j < parameters.numSpots; ++j) {
                if (parameters.spots[j].values.cellColorTransitionTargetColor[i] != i) {
                    parameters.features.cellColorTransitionRules = true;
                    break;
                }
            }
        }
    }

    if (missingFeatures.cellAgeLimiter && parameters.cellMaxAgeBalancer) {
        parameters.features.cellAgeLimiter = true;
    }
}

void LegacySimulationParametersService::activateParametersForLegacyFiles(
    MissingParameters const& missingParameters,
    LegacyParameters const& legacyParameters,
    SimulationParameters& parameters)
{
    if (missingParameters.externalEnergyBackflowFactor) {
        if (!equals(parameters.externalEnergyConditionalInflowFactor, 0.0f)) {
            for (int i = 0; i < MAX_COLORS; ++i) {
                parameters.externalEnergyBackflowFactor[i] = parameters.externalEnergyConditionalInflowFactor[i] / 5;
            }
        }
    }

    if (missingParameters.copyMutations) {
        auto setParametersForSpot = [](SimulationParametersSpotValues& target, LegacyParametersForSpot const& source) {
            for (int i = 0; i < MAX_COLORS; ++i) {
                target.cellCopyMutationNeuronData[i] = source.cellFunctionConstructorMutationNeuronDataProbability[i] * 250;
                target.cellCopyMutationCellProperties[i] = source.cellFunctionConstructorMutationPropertiesProbability[i] * 250;
                target.cellCopyMutationCellFunction[i] = source.cellFunctionConstructorMutationCellFunctionProbability[i] * 250;
                target.cellCopyMutationGeometry[i] = source.cellFunctionConstructorMutationGeometryProbability[i] * 250;
                target.cellCopyMutationCustomGeometry[i] = source.cellFunctionConstructorMutationCustomGeometryProbability[i] * 250;
                target.cellCopyMutationInsertion[i] = source.cellFunctionConstructorMutationInsertionProbability[i] * 250;
                target.cellCopyMutationDeletion[i] = source.cellFunctionConstructorMutationDeletionProbability[i] * 250;
                target.cellCopyMutationCellColor[i] = source.cellFunctionConstructorMutationCellColorProbability[i] * 250;
                target.cellCopyMutationTranslation[i] = source.cellFunctionConstructorMutationTranslationProbability[i] * 5000;
                target.cellCopyMutationDuplication[i] = source.cellFunctionConstructorMutationDuplicationProbability[i] * 5000;
                target.cellCopyMutationSubgenomeColor[i] = source.cellFunctionConstructorMutationSubgenomeColorProbability[i] * 5000;
                target.cellCopyMutationGenomeColor[i] = source.cellFunctionConstructorMutationGenomeColorProbability[i] * 5000;
            }
        };
        setParametersForSpot(parameters.baseValues, legacyParameters.base);
        for (int i = 0; i < MAX_SPOTS; ++i) {
            setParametersForSpot(parameters.spots->values, legacyParameters.spots[i]);
        }
    }
}
