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
        auto numSpots = std::min(currentParameters.numLayers, changedParameters.numLayers);
        for (int i = 0; i < numSpots; ++i) {
            if (currentParameters.layerVelocity.layerValues[i].x != 0) {
                result.layerPosition.layerValues[i].x = currentParameters.layerPosition.layerValues[i].x;
            }
            if (currentParameters.layerVelocity.layerValues[i].y != 0) {
                result.layerPosition.layerValues[i].y = currentParameters.layerPosition.layerValues[i].y;
            }
        }

        auto numRadiationSources = std::min(currentParameters.numLayers, changedParameters.numLayers);
        for (int i = 0; i < numRadiationSources; ++i) {
            if (currentParameters.sourceVelocity.sourceValues[i].x != 0) {
                result.sourcePosition.sourceValues[i].x = currentParameters.sourcePosition.sourceValues[i].x;
            }
            if (currentParameters.sourceVelocity.sourceValues[i].y != 0) {
                result.sourcePosition.sourceValues[i].y = currentParameters.sourcePosition.sourceValues[i].y;
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
    for (int i = 0; i < settings.simulationParameters.numSources; ++i) {
        auto& sourcePos = settings.simulationParameters.sourcePosition.sourceValues[i];
        auto& sourceVel = settings.simulationParameters.sourceVelocity.sourceValues[i];
        if (abs(sourceVel.x) > NEAR_ZERO) {
            sourcePos.x += sourceVel.x * settings.simulationParameters.timestepSize.value;
            result = true;
        }
        if (abs(sourceVel.y) > NEAR_ZERO) {
            sourcePos.y += sourceVel.y * settings.simulationParameters.timestepSize.value;
            result = true;
        }
        auto correctedPosition = space.getCorrectedPosition(sourcePos);
        sourcePos = correctedPosition;
    }
    for (int i = 0; i < settings.simulationParameters.numLayers; ++i) {
        auto& layerPosition = settings.simulationParameters.layerPosition.layerValues[i];
        auto& layerVelocity = settings.simulationParameters.layerVelocity.layerValues[i];
        if (abs(layerVelocity.x) > NEAR_ZERO) {
            layerPosition.x += layerVelocity.x * settings.simulationParameters.timestepSize.value;
            result = true;
        }
        if (abs(layerVelocity.y) > NEAR_ZERO) {
            layerPosition.y += layerVelocity.y * settings.simulationParameters.timestepSize.value;
            result = true;
        }
        layerPosition = space.getCorrectedPosition(layerPosition);
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
