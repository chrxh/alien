#include "SimulationParametersEditService.h"

#include "Base/Definitions.h"

auto SimulationParametersEditService::getRadiationStrengths(SimulationParameters const& parameters) const -> RadiationStrengths
{
    RadiationStrengths result;
    result.values.reserve(parameters.numRadiationSources + 1);

    auto baseStrength = 1.0f;
    for (int i = 0; i < parameters.numRadiationSources; ++i) {
        baseStrength -= parameters.radiationSources[i].strengthRatio;
    }
    if (baseStrength < 0) {
        baseStrength = 0;
    }

    result.values.emplace_back(baseStrength);
    for (int i = 0; i < parameters.numRadiationSources; ++i) {
        result.values.emplace_back(parameters.radiationSources[i].strengthRatio);
        if (parameters.radiationSources[i].strengthRatioPinned) {
            result.pinned.insert(i + 1);
        }
    }
    if (parameters.baseStrengthRatioPinned) {
        result.pinned.insert(0);
    }
    return result;
}

void SimulationParametersEditService::applyRadiationStrengths(SimulationParameters& parameters, RadiationStrengths const& ratios)
{
    CHECK(parameters.numRadiationSources + 1 == ratios.values.size());

    for (int i = 0; i < parameters.numRadiationSources; ++i) {
        parameters.radiationSources[i].strengthRatio = ratios.values.at(i + 1);
    }
}

void SimulationParametersEditService::adaptRadiationStrengths(RadiationStrengths& strengths, RadiationStrengths& origStrengths, int changeIndex) const
{
    if (strengths.values.size() == strengths.pinned.size()) {
        strengths = origStrengths;
        return;
    }

    auto sum = 0.0f;
    for (auto const& ratio : strengths.values) {
        sum += ratio;
    }
    auto diff = sum - 1;
    auto sumWithoutFixed = 0.0f;
    for (int i = 0; i < strengths.values.size(); ++i) {
        if (!strengths.pinned.contains(i)) {
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
            if (!strengths.pinned.contains(i)) {
                strengths.values.at(i) *= reduction;
            }
        }
    } else {
        for (int i = 0; i < strengths.values.size(); ++i) {
            if (!strengths.pinned.contains(i)) {
                strengths.values.at(i) = -diff / toFloat(strengths.values.size() - strengths.pinned.size());
            }
        }
    }
    for (auto& ratio : strengths.values) {
        ratio = std::min(1.0f, std::max(0.0f, ratio));
    }
}

auto SimulationParametersEditService::calcRadiationStrengthsForAddingSpot(RadiationStrengths const& strengths) const -> RadiationStrengths
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

auto SimulationParametersEditService::calcRadiationStrengthsForDeletingSpot(
    RadiationStrengths const& strengths, int deleteIndex) const -> RadiationStrengths
{
    auto numRemainingUnpinned = 0;
    auto sumRemainingUnpinnedStrengths = 0.0f;
    for (int i = 0; i < strengths.values.size(); ++i) {
        if (!strengths.pinned.contains(i) && i != deleteIndex) {
            ++numRemainingUnpinned;
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

    return result;
}
