#include "ParametersValidationService.h"

void ParametersValidationService::validateAndCorrect(SimulationParameters& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        parameters.minCellEnergy.baseValue[i] = std::min(parameters.minCellEnergy.baseValue[i], parameters.normalCellEnergy.value[i] * 0.95f);
        parameters.particleSplitEnergy.value[i] = std::max(0.0f, parameters.particleSplitEnergy.value[i]);
        parameters.defenderAntiAttackerStrength.value[i] = std::max(0.0f, parameters.defenderAntiAttackerStrength.value[i]);
        parameters.defenderAntiInjectorStrength.value[i] = std::max(0.0f, parameters.defenderAntiInjectorStrength.value[i]);
    }
    parameters.timestepSize.value = std::max(0.01f, parameters.timestepSize.value);

    for (int i = 0; i < parameters.numZones.value; ++i) {
        validateAndCorrect(parameters.zone[i], parameters);
    }
}

void ParametersValidationService::validateAndCorrect(RadiationSource& source) const
{
    if (source.shape.type == RadiationSourceShapeType_Circular) {
        source.shape.alternatives.circularRadiationSource.radius = std::max(1.0f, source.shape.alternatives.circularRadiationSource.radius);
    }
    if (source.shape.type == RadiationSourceShapeType_Rectangular) {
        source.shape.alternatives.rectangularRadiationSource.width = std::max(1.0f, source.shape.alternatives.rectangularRadiationSource.width);
        source.shape.alternatives.rectangularRadiationSource.height = std::max(1.0f, source.shape.alternatives.rectangularRadiationSource.height);
    }
}

void ParametersValidationService::validateAndCorrect(SimulationParametersZone& zone, SimulationParameters const& parameters) const
{
}
