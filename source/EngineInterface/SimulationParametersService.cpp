#include "SimulationParametersService.h"

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

void SimulationParametersService::activateFeaturesBasedOnParameters(Features const& missingFeatures, SimulationParameters& parameters)
{
    if (missingFeatures.advancedAbsorptionControl) {
        if (!equals(parameters.radiationAbsorptionVelocityPenalty, 0.0f) || !equals(parameters.radiationAbsorptionLowConnectionPenalty, 0.0f)) {
            parameters.features.advancedAbsorptionControl = true;
        }
    }

    if (missingFeatures.advancedAttackerControl) {
        if (!equals(parameters.cellFunctionAttackerGenomeComplexityBonus, 0.0f) || !equals(parameters.cellFunctionAttackerSameMutantPenalty, 0.0f)
            || !equals(parameters.cellFunctionAttackerSensorDetectionFactor, 0.0f)
            || !equals(parameters.baseValues.cellFunctionAttackerGeometryDeviationExponent, 0.0f)
            || !equals(parameters.baseValues.cellFunctionAttackerConnectionsMismatchPenalty, 0.0f)) {
            parameters.features.advancedAttackerControl = true;
        }
    }

    if (missingFeatures.externalEnergyControl) {
        if (!equals(parameters.cellFunctionConstructorExternalEnergy, 0.0f) || !equals(parameters.cellFunctionConstructorExternalEnergySupplyRate, 0.0f)) {
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
}
