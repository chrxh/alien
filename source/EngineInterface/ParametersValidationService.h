#pragma once

#include <string>

#include "Base/Singleton.h"

#include "Definitions.h"
#include "SimulationParameters.h"


class ParametersValidationService
{
    MAKE_SINGLETON(ParametersValidationService);

public:
    struct ValidationConfig
    {
        IntVector2D worldSize;
    };
    void validateAndCorrect(ValidationConfig const& config, SimulationParameters& parameters) const;

private:
    void validateAndCorrectIntern(
        ValidationConfig const& config,
        std::vector<ParameterSpec> const& parameterSpecs,
        SimulationParameters& parameters,
        int orderNumber) const;
};
