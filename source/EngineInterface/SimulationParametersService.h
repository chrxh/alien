#pragma once
#include "SimulationParameters.h"

class SimulationParametersService
{
public:
    static void activateFeaturesBasedOnParameters(Features const& missingFeatures, SimulationParameters& parameters);
    static void activateParametersBasedOnMissingFeatures(Features const& missingFeatures, SimulationParameters& parameters);
};
