#include "SimulationParametersSpecificationService.h"

#include "SimulationParameters.h"

#define BASE_OFFSET(X) offsetof(SimulationParameters, X)

ParametersSpec SimulationParametersSpecificationService::createParametersSpec() const
{
    return ParametersSpec().groups(
        {ParameterGroupSpec()
             .name("Visualization")
            .parameters({
                FloatParameterSpec()
                    .name("Cell radius")
                    .valueAddress(BASE_OFFSET(cellRadius))
                    .min(0.0f)
                    .max(0.5f)
                    .tooltip("Specifies the radius of the drawn cells in unit length."),
                FloatParameterSpec()
                    .name("Zoom level for neural activity")
                    .valueAddress(BASE_OFFSET(zoomLevelForNeuronVisualization))
                    .min(0.0f)
                    .max(32.0f)
                    .infinity(true)
                    .tooltip("The zoom level from which the neuronal activities become visible."),
                BoolParameterSpec()
                    .name("Attack visualization")
                    .valueAddress(BASE_OFFSET(attackVisualization))
                    .tooltip("If activated, successful attacks of attacker cells are visualized."),
                BoolParameterSpec()
                    .name("Muscle movement visualization")
                    .valueAddress(BASE_OFFSET(muscleMovementVisualization))
                    .tooltip("If activated, the direction in which muscle cells are moving are visualized."),
                BoolParameterSpec()
                    .name("Borderless rendering")
                    .valueAddress(BASE_OFFSET(borderlessRendering))
                    .tooltip("If activated, the simulation is rendered periodically in the view port."),
                BoolParameterSpec()
                    .name("Grid lines")
                    .valueAddress(BASE_OFFSET(gridLines))
                    .tooltip("This option draws a suitable grid in the background depending on the zoom level."),
                BoolParameterSpec()
                    .name("Mark reference domain")
                    .valueAddress(BASE_OFFSET(markReferenceDomain))
                    .tooltip("This option draws a suitable grid in the background depending on the zoom level."),
                BoolParameterSpec()
                    .name("Show radiation sources")
                    .valueAddress(BASE_OFFSET(showRadiationSources))
                    .tooltip("This option draws red crosses in the center of radiation sources."),
            }),
        ParameterGroupSpec()
            .name("Numerics")
            .parameters({
                FloatParameterSpec()
                    .name("Time step size")
                    .valueAddress(BASE_OFFSET(timestepSize))
                    .min(0.0f)
                    .max(1.0f)
                    .tooltip("The time duration calculated in a single simulation step. Smaller values increase the accuracy of the simulation while larger "
                             "values can lead to numerical instabilities."),
            }),
    });
}

float& SimulationParametersSpecificationService::getValueRef(FloatParameterSpec const& spec, SimulationParameters& parameters) const
{
    return *(reinterpret_cast<float*>(reinterpret_cast<char*>(&parameters) + spec._valueAddress));
}

bool& SimulationParametersSpecificationService::getValueRef(BoolParameterSpec const& spec, SimulationParameters& parameters) const
{
    return *(reinterpret_cast<bool*>(reinterpret_cast<char*>(&parameters) + spec._valueAddress));
}
