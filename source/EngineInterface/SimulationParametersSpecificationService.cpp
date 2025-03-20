#include "SimulationParametersSpecificationService.h"

#include "SimulationParameters.h"

#define BASE_OFFSET(X) offsetof(SimulationParameters, X)

ParametersSpec SimulationParametersSpecificationService::createParametersSpec() const
{
    return ParametersSpec().groups(
        {ParameterGroupSpec()
             .name("Visualization")
            .parameters({
                FloatParameterSpec().name("Cell radius").valueAddress(BASE_OFFSET(cellRadius)).min(0.0f).max(0.5f),
                FloatParameterSpec()
                    .name("Zoom level for neural activity")
                    .valueAddress(BASE_OFFSET(zoomLevelForNeuronVisualization))
                    .min(0.0f)
                    .max(32.0f)
                    .infinity(true),
            }),
        ParameterGroupSpec()
            .name("Numerics")
            .parameters({
                FloatParameterSpec().name("Time step size").valueAddress(BASE_OFFSET(timestepSize)).min(0.0f).max(1.0f),
            }),
    });
}

float& SimulationParametersSpecificationService::getValueRef(FloatParameterSpec const& spec, SimulationParameters& parameters) const
{
    return *(reinterpret_cast<float*>(reinterpret_cast<char*>(&parameters) + spec._valueAddress));
}
