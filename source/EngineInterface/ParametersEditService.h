#pragma once

#include <set>
#include <string>
#include <vector>

#include "Base/Singleton.h"
#include "SimulationParameters.h"
#include "SimulationParametersSpecification.h"

struct RadiationStrengths
{
    std::vector<float> values;
    std::set<int> pinned;
};

class ParametersEditService
{
    MAKE_SINGLETON(ParametersEditService);

public:
    void cloneLocation(SimulationParameters& parameters, int locationIndex) const;
    void copyLocation(SimulationParameters& parameters, int sourceLocationIndex, int targetLocationIndex) const;

    RadiationStrengths getRadiationStrengths(SimulationParameters const& parameters) const;
    void applyRadiationStrengths(SimulationParameters& parameters, RadiationStrengths const& strengths);

    void adaptRadiationStrengths(RadiationStrengths& strengths, RadiationStrengths& origStrengths, int changeIndex) const;
    RadiationStrengths calcRadiationStrengthsForAddingZone(RadiationStrengths const& strengths) const;
    RadiationStrengths calcRadiationStrengthsForDeletingZone(RadiationStrengths const& strengths, int deleteIndex) const;

private:
    void copyLocationImpl(SimulationParameters& parameters, std::vector<ParameterSpec> const& parameterSpecs, int sourceLocationIndex, int targetLocationIndex) const;
};
