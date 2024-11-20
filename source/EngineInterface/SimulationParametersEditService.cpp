#include "SimulationParametersEditService.h"

#include "Base/Definitions.h"

auto SimulationParametersEditService::getRadiationStrengths(SimulationParameters const& parameters) const -> RadiationStrengths
{
    RadiationStrengths result;
    result.values.reserve(parameters.numRadiationSources + 1);

    auto baseStrengthRatio = 1.0f;
    for (int i = 0; i < parameters.numRadiationSources; ++i) {
        baseStrengthRatio -= parameters.radiationSources[i].strengthRatio;
    }
    if (baseStrengthRatio < 0) {
        baseStrengthRatio = 0;
    }

    result.values.emplace_back(baseStrengthRatio);
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

void SimulationParametersEditService::adaptRadiationStrengths(RadiationStrengths& ratios, RadiationStrengths& origRatios, int changeIndex) const
{
    if (ratios.values.size() == ratios.pinned.size()) {
        ratios = origRatios;
        return;
    }

    auto sum = 0.0f;
    for (auto const& ratio : ratios.values) {
        sum += ratio;
    }
    auto diff = sum - 1;
    auto sumWithoutFixed = 0.0f;
    for (int i = 0; i < ratios.values.size(); ++i) {
        if (!ratios.pinned.contains(i)) {
            sumWithoutFixed += ratios.values.at(i);
        }
    }

    if (sumWithoutFixed < diff) {
        ratios.values.at(changeIndex) -= diff - sumWithoutFixed;
        diff = sumWithoutFixed;
    }
    if (sumWithoutFixed != 0) {
        auto reduction = 1.0f - diff / sumWithoutFixed;

        for (int i = 0; i < ratios.values.size(); ++i) {
            if (!ratios.pinned.contains(i)) {
                ratios.values.at(i) *= reduction;
            }
        }
    } else {
        for (int i = 0; i < ratios.values.size(); ++i) {
            if (!ratios.pinned.contains(i)) {
                ratios.values.at(i) = -diff / toFloat(ratios.values.size() - ratios.pinned.size());
            }
        }
    }
    for (auto& ratio : ratios.values) {
        ratio = std::min(1.0f, std::max(0.0f, ratio));
    }
}

auto SimulationParametersEditService::calcRadiationStrengthsForAddingSpot(RadiationStrengths const& ratios) const -> RadiationStrengths
{
    auto result = ratios;
    if (ratios.values.size() == ratios.pinned.size()) {
        result.values.emplace_back(0.0f);
        return result;
    }

    auto reductionFactor = 1.0f / toFloat(ratios.values.size() - ratios.pinned.size() + 1);
    auto newRatio = 0.0f;

    for (int i = 0; i < ratios.values.size(); ++i) {
        if (!ratios.pinned.contains(i)) {
            newRatio += ratios.values.at(i) * reductionFactor;
            result.values.at(i) = ratios.values.at(i) * (1.0f - reductionFactor);
        }
    }
    result.values.emplace_back(newRatio);
    return result;
}
