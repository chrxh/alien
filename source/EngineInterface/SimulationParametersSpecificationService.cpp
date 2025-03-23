#include "SimulationParametersSpecificationService.h"

#include "SimulationParameters.h"

#define BASE_VALUE_OFFSET(X) offsetof(SimulationParameters, X)
#define ZONE_VALUE_OFFSET(X) offsetof(SimulationParametersZoneValues, X)

ParametersSpec SimulationParametersSpecificationService::createParametersSpec() const
{
    return ParametersSpec().groups({
        ParameterGroupSpec().name("General").parameters({
            ParameterSpec().name("Project name").valueAddress(BASE_VALUE_OFFSET(projectName)).type(Char64Spec()),
            }),
        ParameterGroupSpec()
             .name("Visualization")
            .parameters({
                ParameterSpec()
                    .name("Cell radius")
                    .valueAddress(BASE_VALUE_OFFSET(cellRadius))
                    .tooltip("Specifies the radius of the drawn cells in unit length.")
                    .type(FloatSpec().min(0.0f).max(0.5f)),
                ParameterSpec()
                    .name("Zoom level for neural activity")
                    .valueAddress(BASE_VALUE_OFFSET(zoomLevelForNeuronVisualization))
                    .tooltip("The zoom level from which the neuronal activities become visible.")
                    .type(FloatSpec().min(0.0f).max(32.f).infinity(true)),
                ParameterSpec()
                    .name("Attack visualization")
                    .valueAddress(BASE_VALUE_OFFSET(attackVisualization))
                    .tooltip("If activated, successful attacks of attacker cells are visualized.")
                    .type(BoolSpec()),
                ParameterSpec()
                    .name("Muscle movement visualization")
                    .valueAddress(BASE_VALUE_OFFSET(muscleMovementVisualization))
                    .tooltip("If activated, the direction in which muscle cells are moving are visualized.")
                    .type(BoolSpec()),
                ParameterSpec()
                    .name("Borderless rendering")
                    .valueAddress(BASE_VALUE_OFFSET(borderlessRendering))
                    .tooltip("If activated, the simulation is rendered periodically in the view port.")
                    .type(BoolSpec()),
                ParameterSpec()
                    .name("Grid lines")
                    .valueAddress(BASE_VALUE_OFFSET(gridLines))
                    .tooltip("This option draws a suitable grid in the background depending on the zoom level.")
                    .type(BoolSpec()),
                ParameterSpec()
                    .name("Mark reference domain")
                    .valueAddress(BASE_VALUE_OFFSET(markReferenceDomain))
                    .tooltip("This option draws a suitable grid in the background depending on the zoom level.")
                    .type(BoolSpec()),
                ParameterSpec()
                    .name("Show radiation sources")
                    .valueAddress(BASE_VALUE_OFFSET(showRadiationSources))
                    .tooltip("This option draws red crosses in the center of radiation sources.")
                    .type(BoolSpec()),
            }),
        ParameterGroupSpec()
            .name("Numerics")
            .parameters({
                ParameterSpec()
                    .name("Time step size")
                    .valueAddress(BASE_VALUE_OFFSET(timestepSize))
                    .tooltip("The time duration calculated in a single simulation step. Smaller values increase the accuracy of the simulation while larger "
                             "values can lead to numerical instabilities.")
                    .type(FloatSpec().min(0.0f).max(1.0f)),
            }),
        ParameterGroupSpec()
            .name("Physics: Motion")
            .parameters({
                ParameterSpec()
                    .name("Friction")
                    .valueAddress(ZONE_VALUE_OFFSET(friction))
                    .visibleInZone(true)
                    .tooltip("This specifies the fraction of the velocity that is slowed down per time step.")
                    .type(FloatSpec().min(0.0f).max(1.0f).format("%.4f").logarithmic(true)),
            }),
    });
}

template float& SimulationParametersSpecificationService::getValueRef<float>(ParameterSpec const& spec, SimulationParameters& parameters, int locationIndex)
    const;
template bool& SimulationParametersSpecificationService::getValueRef<bool>(ParameterSpec const& spec, SimulationParameters& parameters, int locationIndex)
    const;
template Char64& SimulationParametersSpecificationService::getValueRef<Char64>(ParameterSpec const& spec, SimulationParameters& parameters, int locationIndex)
    const;

template <typename T>
T& SimulationParametersSpecificationService::getValueRef(ParameterSpec const& spec, SimulationParameters& parameters, int locationIndex) const
{
    if (spec._visibleInBase && !spec._visibleInZone && !spec._visibleInSource) {
        return *(reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters) + spec._valueAddress));
    } else if (spec._visibleInBase && spec._visibleInZone && !spec._visibleInSource) {
        if (locationIndex == 0) {
            return *(reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters.baseValues) + spec._valueAddress));
        }
        for (int i = 0; i < parameters.numZones; ++i) {
            if (parameters.zone[i].locationIndex == locationIndex) {
                return *(reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters.zone[i].values) + spec._valueAddress));
            }
        }
    }
    CHECK(false);
}