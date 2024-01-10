#include "SimulationParametersService.h"

void SimulationParametersService::activateFeaturesBasedOnParameters(SimulationParameters& parameters)
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        if (parameters.cellFunctionConstructorExternalEnergy[i] != 0 || parameters.cellFunctionConstructorExternalEnergySupplyRate[i] != 0) {
            parameters.features.externalEnergy = true;
            break;
        }
    }
    for (int i = 0; i < MAX_COLORS; ++i) {
        if ((parameters.baseValues.cellColorTransitionDuration[i] != Infinity<int>::value
             && parameters.baseValues.cellColorTransitionDuration[i] != 0) || parameters.baseValues.cellColorTransitionTargetColor[i] != i)
            {
            parameters.features.colorTransitions = true;
            break;
        }
    }
}

void SimulationParametersService::resetParametersBasedOnFeatures(SimulationParameters& parameters)
{
    if (!parameters.features.externalEnergy) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            parameters.cellFunctionConstructorExternalEnergy[i] = 0;
            parameters.cellFunctionConstructorExternalEnergySupplyRate[i] = 0;
        }
    }
    if (!parameters.features.colorTransitions) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            parameters.baseValues.cellColorTransitionDuration[i] = Infinity<int>::value;
            parameters.baseValues.cellColorTransitionTargetColor[i] = 0;
        }
    }
}
