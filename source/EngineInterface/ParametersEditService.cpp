#include "ParametersEditService.h"

#include <ranges>

#include <Base/Definitions.h>
#include <Base/StringHelper.h>

#include "ObjectColoring.h"
#include "SimulationFacade.h"
#include "SpecificationEvaluationService.h"

void ParametersEditService::insertDefaultLayer(SimulationParameters& parameters, int orderNumber) const
{
    LocationHelper::adaptLocationIndices(parameters, orderNumber + 1, 1);

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

    auto newLayerIndex = LocationHelper::findLocationArrayIndex(parameters, orderNumber + 1);
    StringHelper::copy(parameters.layerName.layerValues[newLayerIndex], sizeof(Char64), LocationHelper::generateLayerName(parameters));
    parameters.layerIds[newLayerIndex] = LocationHelper::generateLocationId(parameters);
}

void ParametersEditService::insertDefaultSource(SimulationParameters& parameters, int orderNumber) const
{
    LocationHelper::adaptLocationIndices(parameters, orderNumber + 1, 1);

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

    auto newSourceIndex = LocationHelper::findLocationArrayIndex(parameters, orderNumber + 1);
    StringHelper::copy(parameters.sourceName.sourceValues[newSourceIndex], sizeof(Char64), LocationHelper::generateSourceName(parameters));
    parameters.sourceIds[newSourceIndex] = LocationHelper::generateLocationId(parameters);
}

void ParametersEditService::initNewLayer(
    SimulationParameters& parameters,
    int orderNumber,
    IntVector2D const& worldSize,
    RealVector2D const& position,
    FloatColorRGB const& backgroundColor) const
{
    auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
    auto minRadius = toFloat(std::min(worldSize.x, worldSize.y)) / 2;
    parameters.backgroundColor.layerValues[index] = {.value = backgroundColor, .enabled = true};
    parameters.layerPosition.layerValues[index] = position;
    parameters.layerCoreRadius.layerValues[index] = minRadius / 3;
    parameters.layerCoreRect.layerValues[index] = {minRadius / 3, minRadius / 3};
    parameters.layerFadeoutRadius.layerValues[index] = minRadius / 5;
}

void ParametersEditService::cloneLocation(SimulationParameters& parameters, int orderNumber) const
{
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
    auto startIndex = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
    LocationHelper::adaptLocationIndices(parameters, orderNumber, 1);

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
    LocationHelper::setLocationId(parameters, orderNumber + 1, LocationHelper::generateLocationId(parameters));
}

void ParametersEditService::deleteLocation(SimulationParameters& parameters, int orderNumber) const
{
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
    auto startIndex = LocationHelper::findLocationArrayIndex(parameters, orderNumber);

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

    LocationHelper::adaptLocationIndices(parameters, orderNumber + 1, -1);
}

void ParametersEditService::moveLocationUpwards(SimulationParameters& parameters, int orderNumber) const
{
    auto sourceLocationType = LocationHelper::getLocationType(orderNumber, parameters);
    auto targetLocationType = LocationHelper::getLocationType(orderNumber - 1, parameters);

    if (sourceLocationType == targetLocationType) {
        SimulationParameters tempParameters;
        if (sourceLocationType == LocationType::Layer) {
            tempParameters.numLayers = 1;
            tempParameters.layerOrderNumbers[0] = 1;

            auto arrayIndex = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
            auto prevOrderNumber = parameters.layerOrderNumbers[arrayIndex - 1];

            copyLocation(tempParameters, 1, parameters, orderNumber);
            copyLocation(parameters, orderNumber, parameters, prevOrderNumber);
            copyLocation(parameters, prevOrderNumber, tempParameters, 1);
        } else {
            tempParameters.numSources = 1;
            tempParameters.sourceOrderNumbers[0] = 1;

            auto arrayIndex = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
            auto prevOrderNumber = parameters.sourceOrderNumbers[arrayIndex - 1];

            copyLocation(tempParameters, 1, parameters, orderNumber);
            copyLocation(parameters, orderNumber, parameters, prevOrderNumber);
            copyLocation(parameters, prevOrderNumber, tempParameters, 1);
        }
    } else {
        LocationHelper::decreaseOrderNumber(parameters, orderNumber);
    }
}

void ParametersEditService::moveLocationDownwards(SimulationParameters& parameters, int orderNumber) const
{
    auto sourceLocationType = LocationHelper::getLocationType(orderNumber, parameters);
    auto targetLocationType = LocationHelper::getLocationType(orderNumber + 1, parameters);

    if (sourceLocationType == targetLocationType) {
        SimulationParameters tempParameters;
        if (sourceLocationType == LocationType::Layer) {
            tempParameters.numLayers = 1;
            tempParameters.layerOrderNumbers[0] = 1;

            auto arrayIndex = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
            auto nextOrderNumber = parameters.layerOrderNumbers[arrayIndex + 1];

            copyLocation(tempParameters, 1, parameters, orderNumber);
            copyLocation(parameters, orderNumber, parameters, nextOrderNumber);
            copyLocation(parameters, nextOrderNumber, tempParameters, 1);
        } else {
            tempParameters.numSources = 1;
            tempParameters.sourceOrderNumbers[0] = 1;

            auto arrayIndex = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
            auto nextOrderNumber = parameters.sourceOrderNumbers[arrayIndex + 1];

            copyLocation(tempParameters, 1, parameters, orderNumber);
            copyLocation(parameters, orderNumber, parameters, nextOrderNumber);
            copyLocation(parameters, nextOrderNumber, tempParameters, 1);
        }
    } else {
        LocationHelper::increaseOrderNumber(parameters, orderNumber);
    }
}

namespace
{
    auto constexpr NumLayerBackgroundColors = 32;

    FloatColorRGB calcBackgroundColorForNewLayer(int numLayers)
    {
        auto hue = toFloat((2 + numLayers) * 8 % NumLayerBackgroundColors) / toFloat(NumLayerBackgroundColors - 1);
        auto rgb = ObjectColoring::hsvToRgb(hue, 0.8f, 0.2f);
        return {toFloat((rgb >> 16) & 0xff) / 255.0f, toFloat((rgb >> 8) & 0xff) / 255.0f, toFloat(rgb & 0xff) / 255.0f};
    }

    void applyParameters(SimulationParameters const& parameters, SimulationParameters const& origParameters)
    {
        _SimulationFacade::get()->setSimulationParameters(parameters);
        _SimulationFacade::get()->setOriginalSimulationParameters(origParameters);
    }
}

std::optional<int> ParametersEditService::insertDefaultLayer(int orderNumber)
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();

    if (parameters.numLayers == MAX_LAYERS) {
        return std::nullopt;
    }

    insertDefaultLayer(parameters, orderNumber);
    insertDefaultLayer(origParameters, orderNumber);

    auto newOrderNumber = orderNumber + 1;
    auto worldSize = _SimulationFacade::get()->getWorldSize();
    auto position = calcPositionForNewLocation(worldSize);
    auto backgroundColor = calcBackgroundColorForNewLayer(parameters.numLayers);
    initNewLayer(parameters, newOrderNumber, worldSize, position, backgroundColor);
    initNewLayer(origParameters, newOrderNumber, worldSize, position, backgroundColor);

    applyParameters(parameters, origParameters);

    ++_insertedLocationCounter;
    return newOrderNumber;
}

std::optional<int> ParametersEditService::insertDefaultSource(int orderNumber)
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();

    if (parameters.numSources == MAX_SOURCES) {
        return std::nullopt;
    }
    auto newStrengths = calcRadiationStrengthsForAddingSource(getRadiationStrengths(parameters));

    insertDefaultSource(parameters, orderNumber);
    insertDefaultSource(origParameters, orderNumber);

    applyRadiationStrengths(parameters, newStrengths);
    applyRadiationStrengths(origParameters, newStrengths);

    auto newOrderNumber = orderNumber + 1;
    auto index = LocationHelper::findLocationArrayIndex(parameters, newOrderNumber);
    parameters.sourcePosition.sourceValues[index] = calcPositionForNewLocation(_SimulationFacade::get()->getWorldSize());
    origParameters.sourcePosition.sourceValues[index] = parameters.sourcePosition.sourceValues[index];

    applyParameters(parameters, origParameters);

    ++_insertedLocationCounter;
    return newOrderNumber;
}

std::optional<int> ParametersEditService::cloneLocation(int orderNumber)
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();

    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
    if ((locationType == LocationType::Layer && parameters.numLayers == MAX_LAYERS)
        || (locationType == LocationType::Source && parameters.numSources == MAX_SOURCES)) {
        return std::nullopt;
    }
    auto newStrengths = calcRadiationStrengthsForAddingSource(getRadiationStrengths(parameters));

    cloneLocation(parameters, orderNumber);
    cloneLocation(origParameters, orderNumber);

    if (locationType == LocationType::Source) {
        applyRadiationStrengths(parameters, newStrengths);
        applyRadiationStrengths(origParameters, newStrengths);
    }

    applyParameters(parameters, origParameters);
    return orderNumber + 1;
}

void ParametersEditService::deleteLocation(int orderNumber)
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();

    deleteLocation(parameters, orderNumber);
    deleteLocation(origParameters, orderNumber);

    applyParameters(parameters, origParameters);
}

void ParametersEditService::moveLocationUpwards(int orderNumber)
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();

    moveLocationUpwards(parameters, orderNumber);
    moveLocationUpwards(origParameters, orderNumber);

    applyParameters(parameters, origParameters);
}

void ParametersEditService::moveLocationDownwards(int orderNumber)
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();

    moveLocationDownwards(parameters, orderNumber);
    moveLocationDownwards(origParameters, orderNumber);

    applyParameters(parameters, origParameters);
}

auto ParametersEditService::getRadiationStrengths(SimulationParameters const& parameters) const -> RadiationStrengths
{
    RadiationStrengths result;
    result.values.reserve(parameters.numSources + 1);

    auto baseStrength = 1.0f;
    for (int i = 0; i < parameters.numSources; ++i) {
        baseStrength -= parameters.sourceRelativeStrength.sourceValues[i].value;
    }
    if (baseStrength < 0) {
        baseStrength = 0;
    }

    result.values.emplace_back(baseStrength);
    for (int i = 0; i < parameters.numSources; ++i) {
        result.values.emplace_back(parameters.sourceRelativeStrength.sourceValues[i].value);
        if (parameters.sourceRelativeStrength.sourceValues[i].pinned) {
            result.pinned.insert(i + 1);
        }
    }
    if (parameters.relativeStrengthBasePin.pinned) {
        result.pinned.insert(0);
    }
    return result;
}

void ParametersEditService::applyRadiationStrengths(SimulationParameters& parameters, RadiationStrengths const& strengths)
{
    CHECK(parameters.numSources + 1 == strengths.values.size());

    parameters.relativeStrengthBasePin.pinned = strengths.pinned.contains(0);
    for (int i = 0; i < parameters.numSources; ++i) {
        parameters.sourceRelativeStrength.sourceValues[i].value = strengths.values.at(i + 1);
        parameters.sourceRelativeStrength.sourceValues[i].pinned = strengths.pinned.contains(i + 1);
    }
}

void ParametersEditService::adaptRadiationStrengths(RadiationStrengths& strengths, RadiationStrengths& origStrengths, int changeIndex) const
{
    auto pinnedValues = strengths.pinned;
    pinnedValues.insert(changeIndex);

    if (strengths.values.size() == pinnedValues.size()) {
        strengths = origStrengths;
        return;
    }
    for (auto const& strength : strengths.values) {
        if (strength < 0) {
            strengths = origStrengths;
            return;
        }
    }

    auto sum = 0.0f;
    for (auto const& strength : strengths.values) {
        sum += strength;
    }
    auto diff = sum - 1;
    auto sumWithoutFixed = 0.0f;
    for (int i = 0; i < strengths.values.size(); ++i) {
        if (!pinnedValues.contains(i)) {
            sumWithoutFixed += strengths.values.at(i);
        }
    }

    if (sumWithoutFixed < diff) {
        strengths.values.at(changeIndex) -= diff - sumWithoutFixed;
        diff = sumWithoutFixed;
    }
    if (sumWithoutFixed != 0) {
        auto reduction = 1.0f - diff / sumWithoutFixed;

        for (int i = 0; i < strengths.values.size(); ++i) {
            if (!pinnedValues.contains(i)) {
                strengths.values.at(i) *= reduction;
            }
        }
    } else {
        for (int i = 0; i < strengths.values.size(); ++i) {
            if (!pinnedValues.contains(i)) {
                strengths.values.at(i) = -diff / toFloat(strengths.values.size() - pinnedValues.size());
            }
        }
    }
    for (auto& ratio : strengths.values) {
        ratio = std::min(1.0f, std::max(0.0f, ratio));
    }
}

auto ParametersEditService::calcRadiationStrengthsForAddingSource(RadiationStrengths const& strengths) const -> RadiationStrengths
{
    auto result = strengths;
    if (strengths.values.size() == strengths.pinned.size()) {
        result.values.emplace_back(0.0f);
        return result;
    }

    auto reductionFactor = 1.0f / toFloat(strengths.values.size() - strengths.pinned.size() + 1);
    auto newRatio = 0.0f;

    for (int i = 0; i < strengths.values.size(); ++i) {
        if (!strengths.pinned.contains(i)) {
            newRatio += strengths.values.at(i) * reductionFactor;
            result.values.at(i) = strengths.values.at(i) * (1.0f - reductionFactor);
        }
    }
    result.values.emplace_back(newRatio);
    return result;
}

auto ParametersEditService::calcRadiationStrengthsForDeletingLayer(RadiationStrengths const& strengths, int deleteIndex) const -> RadiationStrengths
{
    auto existsUnpinned = false;
    auto sumRemainingUnpinnedStrengths = 0.0f;
    for (int i = 0; i < strengths.values.size(); ++i) {
        if (!strengths.pinned.contains(i) && i != deleteIndex) {
            existsUnpinned = true;
            sumRemainingUnpinnedStrengths += strengths.values.at(i);
        }
    }
    auto increaseFactor = sumRemainingUnpinnedStrengths != 0 ? 1.0f + strengths.values.at(deleteIndex) / sumRemainingUnpinnedStrengths : 1.0f;

    RadiationStrengths result;
    for (int i = 0; i < strengths.values.size(); ++i) {
        if (i != deleteIndex) {
            if (!strengths.pinned.contains(i)) {
                result.values.emplace_back(strengths.values.at(i) * increaseFactor);
            } else {
                result.values.emplace_back(strengths.values.at(i));
                if (i < deleteIndex) {
                    result.pinned.insert(i);
                } else {
                    result.pinned.insert(i - 1);
                }
            }
        }
    }
    if (!existsUnpinned) {
        result.values.at(0) += strengths.values.at(deleteIndex);
        result.pinned.erase(0);
    }

    return result;
}

void ParametersEditService::copyLocation(
    SimulationParameters& targetParameters,
    int targetOrderNumber,
    SimulationParameters& sourceParameters,
    int sourceOrderNumber) const
{
    auto const& parametersSpecs = SimulationParameters::getSpec();
    for (auto const& groupSpec : parametersSpecs._groups) {
        copyLocationIntern(targetParameters, targetOrderNumber, sourceParameters, sourceOrderNumber, groupSpec._parameters);
    }

    auto targetLocationType = LocationHelper::getLocationType(targetOrderNumber, targetParameters);
    if (targetLocationType != LocationType::Base && targetLocationType == LocationHelper::getLocationType(sourceOrderNumber, sourceParameters)) {
        LocationHelper::setLocationId(targetParameters, targetOrderNumber, LocationHelper::getLocationId(sourceParameters, sourceOrderNumber));
    }
}

void ParametersEditService::copyLocationIntern(
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

RealVector2D ParametersEditService::calcPositionForNewLocation(IntVector2D const& worldSize) const
{
    return {
        toFloat(worldSize.x / 2 + (_insertedLocationCounter % 10) * worldSize.x / 20),
        toFloat(worldSize.y / 2 + (_insertedLocationCounter % 10) * worldSize.y / 20)};
}
