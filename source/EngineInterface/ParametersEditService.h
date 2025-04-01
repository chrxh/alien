#pragma once

#include <set>
#include <string>
#include <vector>

#include "Base/Singleton.h"
#include "SimulationParameters.h"

struct RadiationStrengths
{
    std::vector<float> values;
    std::set<int> pinned;
};

class ParametersEditService
{
    MAKE_SINGLETON(ParametersEditService);

public:
    RadiationStrengths getRadiationStrengths(SimulationParameters const& parameters) const;
    void applyRadiationStrengths(SimulationParameters& parameters, RadiationStrengths const& strengths);

    void adaptRadiationStrengths(RadiationStrengths& strengths, RadiationStrengths& origStrengths, int changeIndex) const;
    RadiationStrengths calcRadiationStrengthsForAddingZone(RadiationStrengths const& strengths) const;
    RadiationStrengths calcRadiationStrengthsForDeletingZone(RadiationStrengths const& strengths, int deleteIndex) const;
};
