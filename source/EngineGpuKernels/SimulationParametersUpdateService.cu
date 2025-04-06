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
        auto numSpots = std::min(currentParameters.numZones.value, changedParameters.numZones.value);
        for (int i = 0; i < numSpots; ++i) {
            if (currentParameters.zoneVelocity.zoneValues[i].x != 0) {
                result.zonePosition.zoneValues[i].x = currentParameters.zonePosition.zoneValues[i].x;
            }
            if (currentParameters.zoneVelocity.zoneValues[i].y != 0) {
                result.zonePosition.zoneValues[i].y = currentParameters.zonePosition.zoneValues[i].y;
            }
        }

        auto numRadiationSources = std::min(currentParameters.numZones.value, changedParameters.numZones.value);
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
    SettingsForSimulation& settings,
    MaxAgeBalancer const& maxAgeBalancer,
    SimulationData const& simulationData,
    StatisticsRawData const& statistics)
{
    auto result = false;

    auto const& worldSizeX = settings.worldSizeX;
    auto const& worldSizeY = settings.worldSizeY;
    SpaceCalculator space({worldSizeX, worldSizeY});
    for (int i = 0; i < settings.simulationParameters.numZones.value; ++i) {
        auto& source = settings.simulationParameters.radiationSource[i];
        if (abs(source.velX) > NEAR_ZERO) {
            source.posX += source.velX * settings.simulationParameters.timestepSize.value;
            result = true;
        }
        if (abs(source.velY) > NEAR_ZERO) {
            source.posY += source.velY * settings.simulationParameters.timestepSize.value;
            result = true;
        }
        auto correctedPosition = space.getCorrectedPosition({source.posX, source.posY});
        source.posX = correctedPosition.x;
        source.posY = correctedPosition.y;
    }
    for (int i = 0; i < settings.simulationParameters.numZones.value; ++i) {
        auto& zonePosition = settings.simulationParameters.zonePosition.zoneValues[i];
        auto& zoneVelocity = settings.simulationParameters.zoneVelocity.zoneValues[i];
        if (abs(zoneVelocity.x) > NEAR_ZERO) {
            zonePosition.x += zoneVelocity.x * settings.simulationParameters.timestepSize.value;
            result = true;
        }
        if (abs(zoneVelocity.y) > NEAR_ZERO) {
            zonePosition.y += zoneVelocity.y * settings.simulationParameters.timestepSize.value;
            result = true;
        }
        zonePosition = space.getCorrectedPosition(zonePosition);
    }

    auto externalEnergyPresent = settings.simulationParameters.externalEnergy.value > 0;
    for (int i = 0; i < MAX_COLORS; ++i) {
        externalEnergyPresent |= settings.simulationParameters.externalEnergyBackflowFactor.value[i] > 0;
    }
    externalEnergyPresent &= settings.simulationParameters.externalEnergyControlToggle.value;
    if (externalEnergyPresent) {
        double temp;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&temp, simulationData.externalEnergy, sizeof(double), cudaMemcpyDeviceToHost));
        settings.simulationParameters.externalEnergy.value = toFloat(temp);
        result = true;
    }

    result |= maxAgeBalancer->balance(settings.simulationParameters, statistics, simulationData.timestep);

    return result;
}
