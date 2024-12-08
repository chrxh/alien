#pragma once

#include <string>

#include "Base/Singleton.h"

#include "SimulationParameters.h"


class SimulationParametersValidationService
{
    MAKE_SINGLETON(SimulationParametersValidationService);

public:
    void validateAndCorrect(SimulationParameters& parameters) const;
    void validateAndCorrect(SimulationParametersZone& zone, SimulationParameters const& parameters) const;
    void validateAndCorrect(RadiationSource& source) const;
};
