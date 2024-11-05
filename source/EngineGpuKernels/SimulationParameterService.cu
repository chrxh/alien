#include "SimulationParameterService.cuh"

#include <vector>

#include "EngineInterface/SimulationParameters.h"

SimulationParameters SimulationParameterService::integrateChanges(SimulationParameters const& currentParameters, SimulationParameters const& changedParameters) const
{
    auto result = changedParameters;

    auto numSpots = std::min(currentParameters.numSpots, changedParameters.numSpots);
    for (int i = 0; i < numSpots; ++i) {
        if (currentParameters.spots[i].velX != 0) {
            result.spots[i].posX = currentParameters.spots[i].posX;
        }
        if (currentParameters.spots[i].velY != 0) {
            result.spots[i].posY = currentParameters.spots[i].posY;
        }
    }

    auto numRadiationSources = std::min(currentParameters.numRadiationSources, changedParameters.numRadiationSources);
    for (int i = 0; i < numRadiationSources; ++i) {
        if (currentParameters.radiationSources[i].velX != 0) {
            result.radiationSources[i].posX = currentParameters.radiationSources[i].posX;
        }
        if (currentParameters.spots[i].velY != 0) {
            result.radiationSources[i].posY = currentParameters.radiationSources[i].posY;
        }
    }
    return result;
}
