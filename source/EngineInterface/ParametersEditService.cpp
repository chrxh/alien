#include "ParametersEditService.h"

#include "Base/Definitions.h"

auto ParametersEditService::getRadiationStrengths(SimulationParameters const& parameters) const -> RadiationStrengths
{
    RadiationStrengths result;
    result.values.reserve(parameters.numSources.value + 1);

    auto baseStrength = 1.0f;
    for (int i = 0; i < parameters.numSources.value; ++i) {
        baseStrength -= parameters.radiationSource[i].strength;
    }
    if (baseStrength < 0) {
        baseStrength = 0;
    }

    result.values.emplace_back(baseStrength);
    for (int i = 0; i < parameters.numSources.value; ++i) {
        result.values.emplace_back(parameters.radiationSource[i].strength);
        if (parameters.radiationSource[i].strengthPinned) {
            result.pinned.insert(i + 1);
        }
    }
    if (parameters.relativeStrengthPinned.pinned) {
        result.pinned.insert(0);
    }
    return result;
}

void ParametersEditService::applyRadiationStrengths(SimulationParameters& parameters, RadiationStrengths const& strengths)
{
    CHECK(parameters.numSources.value + 1 == strengths.values.size());

    parameters.relativeStrengthPinned.pinned = strengths.pinned.contains(0);
    for (int i = 0; i < parameters.numSources.value; ++i) {
        parameters.radiationSource[i].strength = strengths.values.at(i + 1);
        parameters.radiationSource[i].strengthPinned = strengths.pinned.contains(i + 1);
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
