#pragma once

#include <set>
#include <string>
#include <vector>

#include "Base/Singleton.h"
#include "SimulationParameters.h"


class SimulationParametersEditService
{
    MAKE_SINGLETON(SimulationParametersEditService);

public:
    struct RadiationStrengths
    {
        std::vector<float> values;
        std::set<int> pinned;
    };
    RadiationStrengths getRadiationStrengths(SimulationParameters const& parameters) const;
    void applyRadiationStrengths(SimulationParameters& parameters, RadiationStrengths const& ratios);

    void adaptRadiationStrengths(RadiationStrengths& ratios, RadiationStrengths& origRatios, int changeIndex) const;
    RadiationStrengths calcRadiationStrengthsForAddingSpot(RadiationStrengths const& ratios) const;
};
