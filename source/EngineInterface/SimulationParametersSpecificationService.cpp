#include "SimulationParametersSpecificationService.h"

#include "SimulationParameters.h"

#define BASE_VALUE_OFFSET(X) offsetof(SimulationParameters, X)
#define ZONE_VALUE_OFFSET(X) offsetof(SimulationParametersZoneValues, X)

ParametersSpec SimulationParametersSpecificationService::createParametersSpec() const
{
    return ParametersSpec().groups(
        {ParameterGroupSpec()
             .name("Visualization")
            .parameters({
                FloatParameterSpec()
                    .name("Cell radius")
                    .valueAddress(BASE_VALUE_OFFSET(cellRadius))
                    .min(0.0f)
                    .max(0.5f)
                    .tooltip("Specifies the radius of the drawn cells in unit length."),
                FloatParameterSpec()
                    .name("Zoom level for neural activity")
                    .valueAddress(BASE_VALUE_OFFSET(zoomLevelForNeuronVisualization))
                    .min(0.0f)
                    .max(32.0f)
                    .infinity(true)
                    .tooltip("The zoom level from which the neuronal activities become visible."),
                BoolParameterSpec()
                    .name("Attack visualization")
                    .valueAddress(BASE_VALUE_OFFSET(attackVisualization))
                    .tooltip("If activated, successful attacks of attacker cells are visualized."),
                BoolParameterSpec()
                    .name("Muscle movement visualization")
                    .valueAddress(BASE_VALUE_OFFSET(muscleMovementVisualization))
                    .tooltip("If activated, the direction in which muscle cells are moving are visualized."),
                BoolParameterSpec()
                    .name("Borderless rendering")
                    .valueAddress(BASE_VALUE_OFFSET(borderlessRendering))
                    .tooltip("If activated, the simulation is rendered periodically in the view port."),
                BoolParameterSpec()
                    .name("Grid lines")
                    .valueAddress(BASE_VALUE_OFFSET(gridLines))
                    .tooltip("This option draws a suitable grid in the background depending on the zoom level."),
                BoolParameterSpec()
                    .name("Mark reference domain")
                    .valueAddress(BASE_VALUE_OFFSET(markReferenceDomain))
                    .tooltip("This option draws a suitable grid in the background depending on the zoom level."),
                BoolParameterSpec()
                    .name("Show radiation sources")
                    .valueAddress(BASE_VALUE_OFFSET(showRadiationSources))
                    .tooltip("This option draws red crosses in the center of radiation sources."),
            }),
        ParameterGroupSpec()
            .name("Numerics")
            .parameters({
                FloatParameterSpec()
                    .name("Time step size")
                    .valueAddress(BASE_VALUE_OFFSET(timestepSize))
                    .min(0.0f)
                    .max(1.0f)
                    .tooltip("The time duration calculated in a single simulation step. Smaller values increase the accuracy of the simulation while larger "
                             "values can lead to numerical instabilities."),
            }),
        ParameterGroupSpec()
            .name("Physics: Motion")
            .parameters({
                FloatParameterSpec()
                    .name("Friction")
                    .valueAddress(ZONE_VALUE_OFFSET(friction))
                    .visibleInSpot(true)
                    .min(0.0f)
                    .max(1.0f)
                    .format("%.4f")
                    .logarithmic(true)
                    .tooltip("This specifies the fraction of the velocity that is slowed down per time step."),
            }),
    });
}

float& SimulationParametersSpecificationService::getValueRef(FloatParameterSpec const& spec, SimulationParameters& parameters, int locationIndex) const
{
    if (spec._visibleInBase && !spec._visibleInSpot && !spec._visibleInSource) {
        return *(reinterpret_cast<float*>(reinterpret_cast<char*>(&parameters) + spec._valueAddress));
    }
    if (spec._visibleInBase && spec._visibleInSpot && !spec._visibleInSource) {
        if (locationIndex == 0) {
            return *(reinterpret_cast<float*>(reinterpret_cast<char*>(&parameters.baseValues) + spec._valueAddress));
        }
        for (int i = 0; i < parameters.numZones; ++i) {
            if (parameters.zone[i].locationIndex == locationIndex) {
                return *(reinterpret_cast<float*>(reinterpret_cast<char*>(&parameters.zone[i].values) + spec._valueAddress));
            }
        }
    }

    CHECK(false);
}

bool& SimulationParametersSpecificationService::getValueRef(BoolParameterSpec const& spec, SimulationParameters& parameters, int locationIndex) const
{
    return *(reinterpret_cast<bool*>(reinterpret_cast<char*>(&parameters) + spec._valueAddress));
}
