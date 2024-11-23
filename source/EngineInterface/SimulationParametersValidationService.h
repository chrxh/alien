#pragma once

#include <string>

#include "Base/Singleton.h"

#include "SimulationParameters.h"


class SimulationParametersValidationService
{
    MAKE_SINGLETON(SimulationParametersValidationService);

public:
    void validateAndCorrect(SimulationParameters& parameters) const;

private:
    void validateAndCorrect(SimulationParametersSpot& spot, SimulationParameters const& parameters) const;
};
