#include "LocationEditService.h"

#include <algorithm>
#include <ranges>

#include <Base/Definitions.h>
#include <Base/StringHelper.h>

#include "LocationAccessService.h"
#include "ObjectColoring.h"
#include "RadiationStrengthService.h"
#include "SpecificationEvaluationService.h"

namespace
{
    auto constexpr NumLayerBackgroundColors = 32;

    FloatColorRGB calcBackgroundColorForNewLayer(int numLayers)
    {
        auto hue = toFloat((2 + numLayers) * 8 % NumLayerBackgroundColors) / toFloat(NumLayerBackgroundColors - 1);
        auto rgb = ObjectColoring::hsvToRgb(hue, 0.8f, 0.2f);
        return {toFloat((rgb >> 16) & 0xff) / 255.0f, toFloat((rgb >> 8) & 0xff) / 255.0f, toFloat(rgb & 0xff) / 255.0f};
    }
}

std::optional<int> LocationEditService::insertDefaultLayer(
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int orderNumber,
    IntVector2D const& worldSize) const
{
    if (parameters.numLayers == MAX_LAYERS) {
        return std::nullopt;
    }

    auto locationId = generateLocationId(parameters);
    insertDefaultLayer(parameters, orderNumber, locationId);
    insertDefaultLayer(origParameters, orderNumber, locationId);

    auto newOrderNumber = orderNumber + 1;
    auto position = calcPositionForNewLocation(worldSize);
    auto backgroundColor = calcBackgroundColorForNewLayer(parameters.numLayers);
    initNewLayer(parameters, newOrderNumber, worldSize, position, backgroundColor);
    initNewLayer(origParameters, newOrderNumber, worldSize, position, backgroundColor);

    ++_insertedLocationCounter;
    return newOrderNumber;
}

std::optional<int> LocationEditService::insertDefaultSource(
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int orderNumber,
    IntVector2D const& worldSize) const
{
    if (parameters.numSources == MAX_SOURCES) {
        return std::nullopt;
    }
    auto const& strengthService = RadiationStrengthService::get();
    auto newStrengths = strengthService.calcRadiationStrengthsForAddingSource(strengthService.getRadiationStrengths(parameters));

    auto locationId = generateLocationId(parameters);
    insertDefaultSource(parameters, orderNumber, locationId);
    insertDefaultSource(origParameters, orderNumber, locationId);

    strengthService.applyRadiationStrengths(parameters, newStrengths);
    strengthService.applyRadiationStrengths(origParameters, newStrengths);

    auto newOrderNumber = orderNumber + 1;
    auto index = LocationAccessService::get().findLocationArrayIndex(parameters, newOrderNumber);
    parameters.sourcePosition.sourceValues[index] = calcPositionForNewLocation(worldSize);
    origParameters.sourcePosition.sourceValues[index] = parameters.sourcePosition.sourceValues[index];

    ++_insertedLocationCounter;
    return newOrderNumber;
}

std::optional<int> LocationEditService::cloneLocation(SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber) const
{
    auto locationType = LocationAccessService::get().getLocationType(orderNumber, parameters);
    if ((locationType == LocationType::Layer && parameters.numLayers == MAX_LAYERS)
        || (locationType == LocationType::Source && parameters.numSources == MAX_SOURCES)) {
        return std::nullopt;
    }
    auto const& strengthService = RadiationStrengthService::get();
    auto newStrengths = strengthService.calcRadiationStrengthsForAddingSource(strengthService.getRadiationStrengths(parameters));

    auto locationId = generateLocationId(parameters);
    cloneLocation(parameters, orderNumber, locationId);
    cloneLocation(origParameters, orderNumber, locationId);

    if (locationType == LocationType::Source) {
        strengthService.applyRadiationStrengths(parameters, newStrengths);
        strengthService.applyRadiationStrengths(origParameters, newStrengths);
    }
    return orderNumber + 1;
}

void LocationEditService::deleteLocation(SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber) const
{
    deleteLocation(parameters, orderNumber);
    deleteLocation(origParameters, orderNumber);
}

void LocationEditService::moveLocationUpwards(SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber) const
{
    moveLocationUpwards(parameters, orderNumber);
    moveLocationUpwards(origParameters, orderNumber);
}

void LocationEditService::moveLocationDownwards(SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber) const
{
    moveLocationDownwards(parameters, orderNumber);
    moveLocationDownwards(origParameters, orderNumber);
}

int LocationEditService::generateLocationId(SimulationParameters const& parameters) const
{
    _lastLocationId = std::max(_lastLocationId, LocationAccessService::get().getMaxLocationId(parameters)) + 1;
    return _lastLocationId;
}

void LocationEditService::insertDefaultLayer(SimulationParameters& parameters, int orderNumber, int locationId) const
{
    adaptLocationIndices(parameters, orderNumber + 1, 1);

    auto startIndex = 0;
    auto insertAtEnd = false;
    for (int i = 0; i < parameters.numLayers; ++i) {
        if (parameters.layerOrderNumbers[i] > orderNumber) {
            startIndex = i;
            break;
        }
        if (i == parameters.numLayers - 1) {
            insertAtEnd = true;
            startIndex = i;
        }
    }

    ++parameters.numLayers;
    for (int i = parameters.numLayers - 2; i >= startIndex; --i) {
        parameters.layerOrderNumbers[i + 1] = parameters.layerOrderNumbers[i];
    }
    parameters.layerOrderNumbers[insertAtEnd ? startIndex + 1 : startIndex] = orderNumber + 1;

    for (int i = parameters.numLayers - 2; i >= startIndex; --i) {
        auto sourceOrderNumber = parameters.layerOrderNumbers[i];
        auto targetOrderNumber = parameters.layerOrderNumbers[i + 1];
        copyLocation(parameters, targetOrderNumber, parameters, sourceOrderNumber);
    }

    SimulationParameters defaultParameters;
    defaultParameters.numLayers = 1;
    defaultParameters.layerOrderNumbers[0] = 1;
    copyLocation(parameters, orderNumber + 1, defaultParameters, 1);

    auto newLayerIndex = LocationAccessService::get().findLocationArrayIndex(parameters, orderNumber + 1);
    StringHelper::copy(parameters.layerName.layerValues[newLayerIndex], sizeof(Char64), generateLayerName(parameters));
    parameters.layerIds[newLayerIndex] = locationId;
}

void LocationEditService::insertDefaultSource(SimulationParameters& parameters, int orderNumber, int locationId) const
{
    adaptLocationIndices(parameters, orderNumber + 1, 1);

    auto startIndex = 0;
    auto insertAtEnd = false;
    for (int i = 0; i < parameters.numSources; ++i) {
        if (parameters.sourceOrderNumbers[i] > orderNumber) {
            startIndex = i;
            break;
        }
        if (i == parameters.numSources - 1) {
            insertAtEnd = true;
            startIndex = i;
        }
    }

    ++parameters.numSources;
    for (int i = parameters.numSources - 2; i >= startIndex; --i) {
        parameters.sourceOrderNumbers[i + 1] = parameters.sourceOrderNumbers[i];
    }
    parameters.sourceOrderNumbers[insertAtEnd ? startIndex + 1 : startIndex] = orderNumber + 1;

    for (int i = parameters.numSources - 2; i >= startIndex; --i) {
        auto sourceOrderNumber = parameters.sourceOrderNumbers[i];
        auto targetOrderNumber = parameters.sourceOrderNumbers[i + 1];
        copyLocation(parameters, targetOrderNumber, parameters, sourceOrderNumber);
    }

    SimulationParameters defaultParameters;
    defaultParameters.numSources = 1;
    defaultParameters.sourceOrderNumbers[0] = 1;
    copyLocation(parameters, orderNumber + 1, defaultParameters, 1);

    auto newSourceIndex = LocationAccessService::get().findLocationArrayIndex(parameters, orderNumber + 1);
    StringHelper::copy(parameters.sourceName.sourceValues[newSourceIndex], sizeof(Char64), generateSourceName(parameters));
    parameters.sourceIds[newSourceIndex] = locationId;
}

void LocationEditService::initNewLayer(
    SimulationParameters& parameters,
    int orderNumber,
    IntVector2D const& worldSize,
    RealVector2D const& position,
    FloatColorRGB const& backgroundColor) const
{
    auto index = LocationAccessService::get().findLocationArrayIndex(parameters, orderNumber);
    auto minRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
    parameters.backgroundColor.layerValues[index] = {.value = backgroundColor, .enabled = true};
    parameters.layerPosition.layerValues[index] = position;
    parameters.layerCoreRadius.layerValues[index] = minRadius / 3;
    parameters.layerCoreRect.layerValues[index] = {minRadius / 3, minRadius / 3};
    parameters.layerFadeoutRadius.layerValues[index] = minRadius / 5;
}

void LocationEditService::cloneLocation(SimulationParameters& parameters, int orderNumber, int locationId) const
{
    auto locationType = LocationAccessService::get().getLocationType(orderNumber, parameters);
    auto startIndex = LocationAccessService::get().findLocationArrayIndex(parameters, orderNumber);
    adaptLocationIndices(parameters, orderNumber, 1);

    if (locationType == LocationType::Layer) {
        ++parameters.numLayers;
        for (int i = parameters.numLayers - 2; i >= startIndex; --i) {
            parameters.layerOrderNumbers[i + 1] = parameters.layerOrderNumbers[i];
        }
        parameters.layerOrderNumbers[startIndex] = orderNumber;

        for (int i = parameters.numLayers - 2; i >= startIndex; --i) {
            auto sourceOrderNumber = parameters.layerOrderNumbers[i];
            auto targetOrderNumber = parameters.layerOrderNumbers[i + 1];
            copyLocation(parameters, targetOrderNumber, parameters, sourceOrderNumber);
        }
    } else {
        ++parameters.numSources;
        for (int i = parameters.numSources - 2; i >= startIndex; --i) {
            parameters.sourceOrderNumbers[i + 1] = parameters.sourceOrderNumbers[i];
        }
        parameters.sourceOrderNumbers[startIndex] = orderNumber;

        for (int i = parameters.numSources - 2; i >= startIndex; --i) {
            auto sourceOrderNumber = parameters.sourceOrderNumbers[i];
            auto targetOrderNumber = parameters.sourceOrderNumbers[i + 1];
            copyLocation(parameters, targetOrderNumber, parameters, sourceOrderNumber);
        }
    }
    setLocationId(parameters, orderNumber + 1, locationId);
}

void LocationEditService::deleteLocation(SimulationParameters& parameters, int orderNumber) const
{
    _lastLocationId = std::max(_lastLocationId, LocationAccessService::get().getMaxLocationId(parameters));

    auto locationType = LocationAccessService::get().getLocationType(orderNumber, parameters);
    auto startIndex = LocationAccessService::get().findLocationArrayIndex(parameters, orderNumber);

    if (locationType == LocationType::Layer) {
        for (int i = startIndex; i < parameters.numLayers - 1; ++i) {
            auto targetOrderNumber = parameters.layerOrderNumbers[i];
            auto sourceOrderNumber = parameters.layerOrderNumbers[i + 1];
            copyLocation(parameters, targetOrderNumber, parameters, sourceOrderNumber);
        }
        for (int i = startIndex; i < parameters.numLayers - 1; ++i) {
            parameters.layerOrderNumbers[i] = parameters.layerOrderNumbers[i + 1];
        }
        --parameters.numLayers;
    } else {
        for (int i = startIndex; i < parameters.numSources - 1; ++i) {
            auto targetOrderNumber = parameters.sourceOrderNumbers[i];
            auto sourceOrderNumber = parameters.sourceOrderNumbers[i + 1];
            copyLocation(parameters, targetOrderNumber, parameters, sourceOrderNumber);
        }
        for (int i = startIndex; i < parameters.numSources - 1; ++i) {
            parameters.sourceOrderNumbers[i] = parameters.sourceOrderNumbers[i + 1];
        }
        --parameters.numSources;
    }

    adaptLocationIndices(parameters, orderNumber + 1, -1);
}

void LocationEditService::moveLocationUpwards(SimulationParameters& parameters, int orderNumber) const
{
    auto sourceLocationType = LocationAccessService::get().getLocationType(orderNumber, parameters);
    auto targetLocationType = LocationAccessService::get().getLocationType(orderNumber - 1, parameters);

    if (sourceLocationType == targetLocationType) {
        SimulationParameters tempParameters;
        if (sourceLocationType == LocationType::Layer) {
            tempParameters.numLayers = 1;
            tempParameters.layerOrderNumbers[0] = 1;

            auto arrayIndex = LocationAccessService::get().findLocationArrayIndex(parameters, orderNumber);
            auto prevOrderNumber = parameters.layerOrderNumbers[arrayIndex - 1];

            copyLocation(tempParameters, 1, parameters, orderNumber);
            copyLocation(parameters, orderNumber, parameters, prevOrderNumber);
            copyLocation(parameters, prevOrderNumber, tempParameters, 1);
        } else {
            tempParameters.numSources = 1;
            tempParameters.sourceOrderNumbers[0] = 1;

            auto arrayIndex = LocationAccessService::get().findLocationArrayIndex(parameters, orderNumber);
            auto prevOrderNumber = parameters.sourceOrderNumbers[arrayIndex - 1];

            copyLocation(tempParameters, 1, parameters, orderNumber);
            copyLocation(parameters, orderNumber, parameters, prevOrderNumber);
            copyLocation(parameters, prevOrderNumber, tempParameters, 1);
        }
    } else {
        decreaseOrderNumber(parameters, orderNumber);
    }
}

void LocationEditService::moveLocationDownwards(SimulationParameters& parameters, int orderNumber) const
{
    auto sourceLocationType = LocationAccessService::get().getLocationType(orderNumber, parameters);
    auto targetLocationType = LocationAccessService::get().getLocationType(orderNumber + 1, parameters);

    if (sourceLocationType == targetLocationType) {
        SimulationParameters tempParameters;
        if (sourceLocationType == LocationType::Layer) {
            tempParameters.numLayers = 1;
            tempParameters.layerOrderNumbers[0] = 1;

            auto arrayIndex = LocationAccessService::get().findLocationArrayIndex(parameters, orderNumber);
            auto nextOrderNumber = parameters.layerOrderNumbers[arrayIndex + 1];

            copyLocation(tempParameters, 1, parameters, orderNumber);
            copyLocation(parameters, orderNumber, parameters, nextOrderNumber);
            copyLocation(parameters, nextOrderNumber, tempParameters, 1);
        } else {
            tempParameters.numSources = 1;
            tempParameters.sourceOrderNumbers[0] = 1;

            auto arrayIndex = LocationAccessService::get().findLocationArrayIndex(parameters, orderNumber);
            auto nextOrderNumber = parameters.sourceOrderNumbers[arrayIndex + 1];

            copyLocation(tempParameters, 1, parameters, orderNumber);
            copyLocation(parameters, orderNumber, parameters, nextOrderNumber);
            copyLocation(parameters, nextOrderNumber, tempParameters, 1);
        }
    } else {
        increaseOrderNumber(parameters, orderNumber);
    }
}

std::string LocationEditService::generateLayerName(SimulationParameters const& parameters) const
{
    int counter = 0;
    bool alreadyUsed;
    std::string result;
    do {
        alreadyUsed = false;
        result = "Layer " + std::to_string(counter++);
        for (int i = 0; i < parameters.numLayers; ++i) {
            auto name = std::string(parameters.layerName.layerValues[i]);
            if (result == name) {
                alreadyUsed = true;
                break;
            }
        }
    } while (alreadyUsed);

    return result;
}

std::string LocationEditService::generateSourceName(SimulationParameters const& parameters) const
{
    int counter = 0;
    bool alreadyUsed;
    std::string result;
    do {
        alreadyUsed = false;
        result = "Radiation " + std::to_string(counter++);
        for (int i = 0; i < parameters.numSources; ++i) {
            auto name = std::string(parameters.sourceName.sourceValues[i]);
            if (result == name) {
                alreadyUsed = true;
                break;
            }
        }
    } while (alreadyUsed);

    return result;
}

void LocationEditService::assignLocationIds(SimulationParameters& parameters) const
{
    for (int i = 0; i < parameters.numLayers; ++i) {
        parameters.layerIds[i] = i + 1;
    }
    for (int i = 0; i < parameters.numSources; ++i) {
        parameters.sourceIds[i] = parameters.numLayers + i + 1;
    }
}

int& LocationEditService::findOrderNumberRef(SimulationParameters& parameters, int orderNumber) const
{
    for (int i = 0; i < parameters.numLayers; ++i) {
        if (parameters.layerOrderNumbers[i] == orderNumber) {
            return parameters.layerOrderNumbers[i];
        }
    }
    for (int i = 0; i < parameters.numSources; ++i) {
        if (parameters.sourceOrderNumbers[i] == orderNumber) {
            return parameters.sourceOrderNumbers[i];
        }
    }

    CHECK(false);
}

void LocationEditService::decreaseOrderNumber(SimulationParameters& parameters, int orderNumber) const
{
    auto& orderNumberRef1 = findOrderNumberRef(parameters, orderNumber);
    auto& orderNumberRef2 = findOrderNumberRef(parameters, orderNumber - 1);
    --orderNumberRef1;
    ++orderNumberRef2;
}

void LocationEditService::increaseOrderNumber(SimulationParameters& parameters, int orderNumber) const
{
    auto& orderNumberRef1 = findOrderNumberRef(parameters, orderNumber);
    auto& orderNumberRef2 = findOrderNumberRef(parameters, orderNumber + 1);
    ++orderNumberRef1;
    --orderNumberRef2;
}

void LocationEditService::adaptLocationIndices(SimulationParameters& parameters, int fromOrderNumber, int offset) const
{
    for (int i = 0; i < parameters.numLayers; ++i) {
        auto& orderNumber = parameters.layerOrderNumbers[i];
        if (orderNumber >= fromOrderNumber) {
            orderNumber += offset;
        }
    }
    for (int i = 0; i < parameters.numSources; ++i) {
        auto& orderNumber = parameters.sourceOrderNumbers[i];
        if (orderNumber >= fromOrderNumber) {
            orderNumber += offset;
        }
    }
}

void LocationEditService::setLocationId(SimulationParameters& parameters, int orderNumber, int locationId) const
{
    auto const& accessService = LocationAccessService::get();
    auto locationType = accessService.getLocationType(orderNumber, parameters);
    CHECK(locationType != LocationType::Base);

    auto index = accessService.findLocationArrayIndex(parameters, orderNumber);
    if (locationType == LocationType::Layer) {
        parameters.layerIds[index] = locationId;
    } else {
        parameters.sourceIds[index] = locationId;
    }
}

void LocationEditService::copyLocation(
    SimulationParameters& targetParameters,
    int targetOrderNumber,
    SimulationParameters& sourceParameters,
    int sourceOrderNumber) const
{
    auto const& parametersSpecs = SimulationParameters::getSpec();
    for (auto const& groupSpec : parametersSpecs._groups) {
        copyLocationIntern(targetParameters, targetOrderNumber, sourceParameters, sourceOrderNumber, groupSpec._parameters);
    }

    auto targetLocationType = LocationAccessService::get().getLocationType(targetOrderNumber, targetParameters);
    if (targetLocationType != LocationType::Base && targetLocationType == LocationAccessService::get().getLocationType(sourceOrderNumber, sourceParameters)) {
        setLocationId(targetParameters, targetOrderNumber, LocationAccessService::get().getLocationId(sourceParameters, sourceOrderNumber));
    }
}

void LocationEditService::copyLocationIntern(
    SimulationParameters& targetParameters,
    int targetOrderNumber,
    SimulationParameters& sourceParameters,
    int sourceOrderNumber,
    std::vector<ParameterSpec> const& parameterSpecs) const
{
    auto& evaluationService = SpecificationEvaluationService::get();

    auto copySourceToTarget = [&](auto const& reference, int sourceOrderNumber, int targetOrderNumber) {
        auto source = evaluationService.getRef(reference._member, sourceParameters, sourceOrderNumber);
        auto target = evaluationService.getRef(reference._member, targetParameters, targetOrderNumber);
        if (source.value != nullptr && target.value != nullptr) {
            if constexpr (std::is_same_v<decltype(source.value), Char64*>) {
                for (int i = 0; i < sizeof(Char64); ++i) {
                    (*target.value)[i] = (*source.value)[i];
                }
            } else {
                if (source.colorDependence == ColorDependence::None) {
                    *target.value = *source.value;
                } else if (source.colorDependence == ColorDependence::ColorVector) {
                    for (int i = 0; i < MAX_COLORS; ++i) {
                        target.value[i] = source.value[i];
                    }
                } else if (source.colorDependence == ColorDependence::ColorMatrix) {
                    for (int i = 0; i < MAX_COLORS * MAX_COLORS; ++i) {
                        target.value[i] = source.value[i];
                    }
                }
            }
        }
        if (source.enabled != nullptr && target.enabled != nullptr) {
            *target.enabled = *source.enabled;
        }
        if (source.pinned != nullptr && target.pinned != nullptr) {
            *target.pinned = *source.pinned;
        }
    };
    for (auto const& parameterSpec : parameterSpecs) {
        if (std::holds_alternative<BoolSpec>(parameterSpec._reference)) {
            copySourceToTarget(std::get<BoolSpec>(parameterSpec._reference), sourceOrderNumber, targetOrderNumber);
        } else if (std::holds_alternative<IntSpec>(parameterSpec._reference)) {
            copySourceToTarget(std::get<IntSpec>(parameterSpec._reference), sourceOrderNumber, targetOrderNumber);
        } else if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            copySourceToTarget(std::get<FloatSpec>(parameterSpec._reference), sourceOrderNumber, targetOrderNumber);
        } else if (std::holds_alternative<Float2Spec>(parameterSpec._reference)) {
            copySourceToTarget(std::get<Float2Spec>(parameterSpec._reference), sourceOrderNumber, targetOrderNumber);
        } else if (std::holds_alternative<Char64Spec>(parameterSpec._reference)) {
            copySourceToTarget(std::get<Char64Spec>(parameterSpec._reference), sourceOrderNumber, targetOrderNumber);
        } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._reference)) {
            auto const& altSpec = std::get<AlternativeSpec>(parameterSpec._reference);
            copySourceToTarget(altSpec, sourceOrderNumber, targetOrderNumber);
            for (auto const& parameterSpecs : altSpec._alternatives | std::views::values) {
                copyLocationIntern(targetParameters, targetOrderNumber, sourceParameters, sourceOrderNumber, parameterSpecs);
            }
        } else if (std::holds_alternative<ColorSpec>(parameterSpec._reference)) {
            copySourceToTarget(std::get<ColorSpec>(parameterSpec._reference), sourceOrderNumber, targetOrderNumber);
        } else if (std::holds_alternative<ColorTransitionRulesSpec>(parameterSpec._reference)) {
            copySourceToTarget(std::get<ColorTransitionRulesSpec>(parameterSpec._reference), sourceOrderNumber, targetOrderNumber);
        }
    }
}

RealVector2D LocationEditService::calcPositionForNewLocation(IntVector2D const& worldSize) const
{
    return {
        toFloat(worldSize.x / 2 + (_insertedLocationCounter % 10) * worldSize.x / 20),
        toFloat(worldSize.y / 2 + (_insertedLocationCounter % 10) * worldSize.y / 20)};
}
