#include "SimulationParametersDescriptionService.h"

#include "SimulationParameters.h"

#define BASE_OFFSET(X) offsetof(SimulationParameters, X)

SimulationParametersDescription SimulationParametersDescriptionService::createSimulationParametersDescription() const
{
    SimulationParametersDescription result;
    result.add(SimulationParametersDescription::AddParameters().name("timestepSize").type(ParameterType::Float).offset(BASE_OFFSET(timestepSize)));
    return result;
}
