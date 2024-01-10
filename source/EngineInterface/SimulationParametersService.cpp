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
}

void SimulationParametersService::activateFeaturesBasedOnParameters(SimulationParameters& parameters)
{
    if (!equals(parameters.cellFunctionConstructorExternalEnergy, 0.0f) || !equals(parameters.cellFunctionConstructorExternalEnergySupplyRate, 0.0f)) {
        parameters.features.externalEnergyControl = true;
    }

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

    if (!equals(parameters.radiationAbsorptionVelocityPenalty, 0.0f) || !equals(parameters.radiationAbsorptionLowConnectionPenalty, 0.0f)) {
        parameters.features.additionalAbsorptionControl = true;
    }
}

namespace
{
    template<typename T>
    void reset(SimulationParameters& parameters, ColorVector<T> SimulationParametersSpotValues::*parameter, T const& value)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            (parameters.baseValues.*parameter)[i] = value;
            for (int j = 0; j < parameters.numSpots; ++j) {
                (parameters.spots[j].values.*parameter)[i] = value;
            }
        }
    }

    template <typename T>
    void reset(ColorVector<T>& parameter, T const& value)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            parameter[i] = value;
        }
    }
}

void SimulationParametersService::resetParametersBasedOnFeatures(SimulationParameters& parameters)
{
    if (!parameters.features.cellColorTransitionRules) {
        reset(parameters, &SimulationParametersSpotValues::cellColorTransitionDuration, Infinity<int>::value);
        for (int i = 0; i < MAX_COLORS; ++i) {
            parameters.baseValues.cellColorTransitionTargetColor[i] = i;
            for (int j = 0; j < parameters.numSpots; ++j) {
                parameters.spots[j].values.cellColorTransitionTargetColor[i] = i;
            }
        }
    }
    if (!parameters.features.externalEnergyControl) {
        reset(parameters.cellFunctionConstructorExternalEnergy, 0.0f);
        reset(parameters.cellFunctionConstructorExternalEnergySupplyRate, 0.0f);
    }
    if (!parameters.features.additionalAbsorptionControl) {
        reset(parameters.radiationAbsorptionVelocityPenalty, 0.0f);
        reset(parameters.radiationAbsorptionLowConnectionPenalty, 0.0f);
    }
}
