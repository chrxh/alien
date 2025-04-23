#pragma once

#include <string>

#include "Base/Singleton.h"

#include "SimulationParameters.h"


class ParametersValidationService
{
    MAKE_SINGLETON(ParametersValidationService);

public:
    void validateAndCorrect(SimulationParameters& parameters) const;
    void validateAndCorrect(RadiationSource& source) const;
};
