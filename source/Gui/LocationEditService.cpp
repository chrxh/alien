#include "LocationEditService.h"

#include <EngineInterface/LocationHelper.h>
#include <EngineInterface/ParametersEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "LocationController.h"

namespace
{
    void applyParameters(SimulationParameters const& parameters, SimulationParameters const& origParameters)
    {
        _SimulationFacade::get()->setSimulationParameters(parameters);
        _SimulationFacade::get()->setOriginalSimulationParameters(origParameters);
    }
}

std::optional<int> LocationEditService::insertDefaultLayer(int orderNumber)
{
    auto& editService = ParametersEditService::get();
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();

    if (parameters.numLayers == MAX_LAYERS) {
        return std::nullopt;
    }

    auto newByOldOrderNumber = editService.insertDefaultLayer(parameters, orderNumber);
    editService.insertDefaultLayer(origParameters, orderNumber);

    auto newOrderNumber = orderNumber + 1;
    auto worldSize = _SimulationFacade::get()->getWorldSize();
    auto position = calcPositionForNewLocation();
    auto backgroundColor = _layerColorPalette.getColor((2 + parameters.numLayers) * 8);
    editService.initNewLayer(parameters, newOrderNumber, worldSize, position, backgroundColor);
    editService.initNewLayer(origParameters, newOrderNumber, worldSize, position, backgroundColor);

    applyParameters(parameters, origParameters);

    LocationController::get().remapLocationIndices(newByOldOrderNumber);
    ++_insertedLocationCounter;
    return newOrderNumber;
}

std::optional<int> LocationEditService::insertDefaultSource(int orderNumber)
{
    auto& editService = ParametersEditService::get();
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();

    if (parameters.numSources == MAX_SOURCES) {
        return std::nullopt;
    }
    auto strengths = editService.getRadiationStrengths(parameters);
    auto newStrengths = editService.calcRadiationStrengthsForAddingSource(strengths);

    auto newByOldOrderNumber = editService.insertDefaultSource(parameters, orderNumber);
    editService.insertDefaultSource(origParameters, orderNumber);

    editService.applyRadiationStrengths(parameters, newStrengths);
    editService.applyRadiationStrengths(origParameters, newStrengths);

    auto newOrderNumber = orderNumber + 1;
    auto index = LocationHelper::findLocationArrayIndex(parameters, newOrderNumber);
    parameters.sourcePosition.sourceValues[index] = calcPositionForNewLocation();
    origParameters.sourcePosition.sourceValues[index] = parameters.sourcePosition.sourceValues[index];

    applyParameters(parameters, origParameters);

    LocationController::get().remapLocationIndices(newByOldOrderNumber);
    ++_insertedLocationCounter;
    return newOrderNumber;
}

std::optional<int> LocationEditService::cloneLocation(int orderNumber)
{
    auto& editService = ParametersEditService::get();
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();

    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
    if ((locationType == LocationType::Layer && parameters.numLayers == MAX_LAYERS)
        || (locationType == LocationType::Source && parameters.numSources == MAX_SOURCES)) {
        return std::nullopt;
    }

    auto strengths = editService.getRadiationStrengths(parameters);
    auto newStrengths = editService.calcRadiationStrengthsForAddingSource(strengths);

    auto newByOldOrderNumber = editService.cloneLocation(parameters, orderNumber);
    editService.cloneLocation(origParameters, orderNumber);

    if (locationType == LocationType::Source) {
        editService.applyRadiationStrengths(parameters, newStrengths);
        editService.applyRadiationStrengths(origParameters, newStrengths);
    }

    applyParameters(parameters, origParameters);

    LocationController::get().remapLocationIndices(newByOldOrderNumber);
    return orderNumber + 1;
}

void LocationEditService::deleteLocation(int orderNumber)
{
    auto& editService = ParametersEditService::get();
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();

    LocationController::get().deleteLocationWindow(orderNumber);

    auto newByOldOrderNumber = editService.deleteLocation(parameters, orderNumber);
    editService.deleteLocation(origParameters, orderNumber);

    applyParameters(parameters, origParameters);

    LocationController::get().remapLocationIndices(newByOldOrderNumber);
}

void LocationEditService::moveLocationUpwards(int orderNumber)
{
    auto& editService = ParametersEditService::get();
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();

    auto newByOldOrderNumber = editService.moveLocationUpwards(parameters, orderNumber);
    editService.moveLocationUpwards(origParameters, orderNumber);

    applyParameters(parameters, origParameters);

    LocationController::get().remapLocationIndices(newByOldOrderNumber);
}

void LocationEditService::moveLocationDownwards(int orderNumber)
{
    auto& editService = ParametersEditService::get();
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();

    auto newByOldOrderNumber = editService.moveLocationDownwards(parameters, orderNumber);
    editService.moveLocationDownwards(origParameters, orderNumber);

    applyParameters(parameters, origParameters);

    LocationController::get().remapLocationIndices(newByOldOrderNumber);
}

RealVector2D LocationEditService::calcPositionForNewLocation() const
{
    auto worldSize = _SimulationFacade::get()->getWorldSize();
    return {
        toFloat(worldSize.x / 2 + (_insertedLocationCounter % 10) * worldSize.x / 20),
        toFloat(worldSize.y / 2 + (_insertedLocationCounter % 10) * worldSize.y / 20)};
}
