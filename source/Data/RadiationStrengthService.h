#pragma once

#include <set>
#include <vector>

#include <Base/Singleton.h>

#include "SimulationParameters.h"

struct RadiationStrengths
{
    std::vector<float> values;
    std::set<int> pinned;
};

class RadiationStrengthService
{
    MAKE_SINGLETON(RadiationStrengthService);

public:
    RadiationStrengths getRadiationStrengths(SimulationParameters const& parameters) const;
    void applyRadiationStrengths(SimulationParameters& parameters, RadiationStrengths const& strengths) const;

    void adaptRadiationStrengths(RadiationStrengths& strengths, RadiationStrengths& origStrengths, int changeIndex) const;
    RadiationStrengths calcRadiationStrengthsForAddingSource(RadiationStrengths const& strengths) const;
    RadiationStrengths calcRadiationStrengthsForDeletingLayer(RadiationStrengths const& strengths, int deleteIndex) const;
};
