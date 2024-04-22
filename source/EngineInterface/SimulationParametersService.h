#pragma once
#include "SimulationParameters.h"

struct MissingParameters
{
    bool externalEnergyBackflowFactor = false;
};

class SimulationParametersService
{
public:
    static void activateFeaturesForLegacyFiles(Features const& missingFeatures, SimulationParameters& parameters);
    static void activateParametersForLegacyFiles(MissingParameters const& missingParameters, SimulationParameters& parameters);
};
