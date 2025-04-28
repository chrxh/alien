#include "ParametersEditService.h"

#include <ranges>

#include "Base/Definitions.h"
#include "Base/StringHelper.h"
#include "SpecificationService.h"
#include "SpecificationEvaluationService.h"

void ParametersEditService::cloneLocation(SimulationParameters& parameters, int locationIndex) const
{
    auto locationType = LocationHelper::getLocationType(locationIndex, parameters);

    auto startIndex = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
    LocationHelper::adaptLocationIndices(parameters, locationIndex, 1);

    if (locationType == LocationType::Zone) {
        ++parameters.numZones;
        for (int i = parameters.numZones - 2; i >= startIndex; --i) {
            parameters.zoneLocationIndex[i + 1] = parameters.zoneLocationIndex[i];
        }
        parameters.zoneLocationIndex[startIndex] = locationIndex;

        for (int i = parameters.numZones - 2; i >= startIndex; --i) {
            auto sourceLocationIndex = parameters.zoneLocationIndex[i];
            auto targetLocationIndex = parameters.zoneLocationIndex[i + 1];
            copyLocation(parameters, sourceLocationIndex, targetLocationIndex);
        }
        StringHelper::copy(
            parameters.zoneName.zoneValues[startIndex + 1],
            sizeof(parameters.zoneName.zoneValues[startIndex + 1]),
            LocationHelper::generateZoneName(parameters));
    } else {
        ++parameters.numSources;
        for (int i = parameters.numSources - 2; i >= startIndex; --i) {
            parameters.sourceLocationIndex[i + 1] = parameters.sourceLocationIndex[i];
        }
        parameters.sourceLocationIndex[startIndex] = locationIndex;

        for (int i = parameters.numSources - 2; i >= startIndex; --i) {
            auto sourceLocationIndex = parameters.sourceLocationIndex[i];
            auto targetLocationIndex = parameters.sourceLocationIndex[i + 1];
            copyLocation(parameters, sourceLocationIndex, targetLocationIndex);
        }
        StringHelper::copy(
            parameters.sourceName.sourceValues[startIndex + 1],
            sizeof(parameters.sourceName.sourceValues[startIndex + 1]),
            LocationHelper::generateSourceName(parameters));
    }
}

void ParametersEditService::copyLocation(SimulationParameters& parameters, int sourceLocationIndex, int targetLocationIndex) const
{
    auto const& parametersSpecs = SpecificationService::get().getSpec();
    for (auto const& groupSpec : parametersSpecs._groups) {
        copyLocationImpl(parameters, groupSpec._parameters, sourceLocationIndex, targetLocationIndex);
    }
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

auto ParametersEditService::calcRadiationStrengthsForAddingZone(RadiationStrengths const& strengths) const -> RadiationStrengths
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

auto ParametersEditService::calcRadiationStrengthsForDeletingZone(
    RadiationStrengths const& strengths, int deleteIndex) const -> RadiationStrengths
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

void ParametersEditService::copyLocationImpl(
    SimulationParameters& parameters,
    std::vector<ParameterSpec> const& parameterSpecs,
    int sourceLocationIndex,
    int targetLocationIndex) const
{
    auto& evaluationService = SpecificationEvaluationService::get();

    auto copySourceToTarget = [&evaluationService, &parameters](auto const& reference, int sourceLocationIndex, int targetLocationIndex) {
        auto source = evaluationService.getRef(reference._member, parameters, sourceLocationIndex);
        auto target = evaluationService.getRef(reference._member, parameters, targetLocationIndex);
        if (source.value != nullptr && target.value != nullptr) {
            if constexpr (std::is_same_v<decltype(source.value), Char64*>) {
                for (int i = 0; i < sizeof(Char64); ++i) {
                    (*target.value)[i] = (*source.value)[i];
                }
            } else {
                if (source.valueType == ColorDependence::None) {
                    *target.value = *source.value;
                } else if (source.valueType == ColorDependence::ColorVector) {
                    for (int i = 0; i < MAX_COLORS; ++i) {
                        target.value[i] = source.value[i];
                    }
                } else if (source.valueType == ColorDependence::ColorMatrix) {
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
            copySourceToTarget(std::get<BoolSpec>(parameterSpec._reference), sourceLocationIndex, targetLocationIndex);
        } else if (std::holds_alternative<IntSpec>(parameterSpec._reference)) {
            copySourceToTarget(std::get<IntSpec>(parameterSpec._reference), sourceLocationIndex, targetLocationIndex);
        } else if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            copySourceToTarget(std::get<FloatSpec>(parameterSpec._reference), sourceLocationIndex, targetLocationIndex);
        } else if (std::holds_alternative<Float2Spec>(parameterSpec._reference)) {
            copySourceToTarget(std::get<Float2Spec>(parameterSpec._reference), sourceLocationIndex, targetLocationIndex);
        } else if (std::holds_alternative<Char64Spec>(parameterSpec._reference)) {
            copySourceToTarget(std::get<Char64Spec>(parameterSpec._reference), sourceLocationIndex, targetLocationIndex);
        } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._reference)) {
            auto const& altSpec = std::get<AlternativeSpec>(parameterSpec._reference);
            copySourceToTarget(altSpec, sourceLocationIndex, targetLocationIndex);
            for (auto const& parameterSpecs : altSpec._alternatives | std::views::values) {
                copyLocationImpl(parameters, parameterSpecs, sourceLocationIndex, targetLocationIndex);
            }
        } else if (std::holds_alternative<ColorPickerSpec>(parameterSpec._reference)) {
            copySourceToTarget(std::get<ColorPickerSpec>(parameterSpec._reference), sourceLocationIndex, targetLocationIndex);
        } else if (std::holds_alternative<ColorTransitionRulesSpec>(parameterSpec._reference)) {
            copySourceToTarget(std::get<ColorTransitionRulesSpec>(parameterSpec._reference), sourceLocationIndex, targetLocationIndex);
        }
    }
}
