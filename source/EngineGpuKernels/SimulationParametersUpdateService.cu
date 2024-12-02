#include "SimulationParametersUpdateService.cuh"

#include <vector>

#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SpaceCalculator.h"

#include "Base.cuh"
#include "SimulationData.cuh"
#include "MaxAgeBalancer.cuh"

SimulationParameters SimulationParametersUpdateService::integrateChanges(
    SimulationParameters const& currentParameters,
    SimulationParameters const& changedParameters,
    SimulationParametersUpdateConfig const& updateConfig) const
{
    auto result = changedParameters;

    if (updateConfig == SimulationParametersUpdateConfig::AllExceptChangingPositions) {
        auto numSpots = std::min(currentParameters.numSpots, changedParameters.numSpots);
        for (int i = 0; i < numSpots; ++i) {
            if (currentParameters.spot[i].velX != 0) {
                result.spot[i].posX = currentParameters.spot[i].posX;
            }
            if (currentParameters.spot[i].velY != 0) {
                result.spot[i].posY = currentParameters.spot[i].posY;
            }
        }

        auto numRadiationSources = std::min(currentParameters.numRadiationSources, changedParameters.numRadiationSources);
        for (int i = 0; i < numRadiationSources; ++i) {
            if (currentParameters.radiationSource[i].velX != 0) {
                result.radiationSource[i].posX = currentParameters.radiationSource[i].posX;
            }
            if (currentParameters.radiationSource[i].velY != 0) {
                result.radiationSource[i].posY = currentParameters.radiationSource[i].posY;
            }
        }
    }
    return result;
}

bool SimulationParametersUpdateService::updateSimulationParametersAfterTimestep(
    Settings& settings,
    MaxAgeBalancer const& maxAgeBalancer,
    SimulationData const& simulationData,
    RawStatisticsData const& statistics)
{
    auto result = false;

    auto const& worldSizeX = settings.generalSettings.worldSizeX;
    auto const& worldSizeY = settings.generalSettings.worldSizeY;
    SpaceCalculator space({worldSizeX, worldSizeY});
    for (int i = 0; i < settings.simulationParameters.numRadiationSources; ++i) {
        auto& source = settings.simulationParameters.radiationSource[i];
        if (abs(source.velX) > NEAR_ZERO) {
            source.posX += source.velX * settings.simulationParameters.timestepSize;
            result = true;
        }
        if (abs(source.velY) > NEAR_ZERO) {
            source.posY += source.velY * settings.simulationParameters.timestepSize;
            result = true;
        }
        auto correctedPosition = space.getCorrectedPosition({source.posX, source.posY});
        source.posX = correctedPosition.x;
        source.posY = correctedPosition.y;
    }
    for (int i = 0; i < settings.simulationParameters.numSpots; ++i) {
        auto& spot = settings.simulationParameters.spot[i];
        if (abs(spot.velX) > NEAR_ZERO) {
            spot.posX += spot.velX * settings.simulationParameters.timestepSize;
            result = true;
        }
        if (abs(spot.velY) > NEAR_ZERO) {
            spot.posY += spot.velY * settings.simulationParameters.timestepSize;
            result = true;
        }
        auto correctedPosition = space.getCorrectedPosition({spot.posX, spot.posY});
        spot.posX = correctedPosition.x;
        spot.posY = correctedPosition.y;
    }

    auto externalEnergyPresent = settings.simulationParameters.externalEnergy > 0;
    for (int i = 0; i < MAX_COLORS; ++i) {
        externalEnergyPresent |= settings.simulationParameters.externalEnergyBackflowFactor[i] > 0;
    }
    externalEnergyPresent &= settings.simulationParameters.features.externalEnergyControl;
    if (externalEnergyPresent) {
        double temp;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&temp, simulationData.externalEnergy, sizeof(double), cudaMemcpyDeviceToHost));
        settings.simulationParameters.externalEnergy = toFloat(temp);
        result = true;
    }

    result |= maxAgeBalancer->balance(settings.simulationParameters, statistics, simulationData.timestep);

    return result;
}
