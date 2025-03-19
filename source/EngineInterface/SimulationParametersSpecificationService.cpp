#include "SimulationParametersSpecificationService.h"

#include "SimulationParameters.h"

#define BASE_OFFSET(X) offsetof(SimulationParameters, X)

ParametersSpec SimulationParametersSpecificationService::createParametersSpec() const
{
    ParametersSpec result;
    //result.add(SimulationParametersSpecification::AddParameters().name("timestepSize").type(ParameterType::Float).offset(BASE_OFFSET(timestepSize)));
    return result;
}
