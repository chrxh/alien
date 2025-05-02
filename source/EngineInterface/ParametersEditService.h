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

using NewByOldLocationIndex = std::map<int, int>;

class ParametersEditService
{
    MAKE_SINGLETON(ParametersEditService);

public:
    NewByOldLocationIndex insertDefaultZone(SimulationParameters& parameters, int locationIndex) const;  // Create location at locationIndex + 1
    NewByOldLocationIndex insertDefaultSource(SimulationParameters& parameters, int locationIndex) const;  // Create location at locationIndex + 1
    NewByOldLocationIndex cloneLocation(SimulationParameters& parameters, int locationIndex) const;        // Create location at locationIndex + 1
    NewByOldLocationIndex deleteLocation(SimulationParameters& parameters, int locationIndex) const;

    RadiationStrengths getRadiationStrengths(SimulationParameters const& parameters) const;
    void applyRadiationStrengths(SimulationParameters& parameters, RadiationStrengths const& strengths);

    void adaptRadiationStrengths(RadiationStrengths& strengths, RadiationStrengths& origStrengths, int changeIndex) const;
    RadiationStrengths calcRadiationStrengthsForAddingSource(RadiationStrengths const& strengths) const;
    RadiationStrengths calcRadiationStrengthsForDeletingZone(RadiationStrengths const& strengths, int deleteIndex) const;

private:
    void copyLocation(SimulationParameters& targetParameters, int targetLocationIndex, SimulationParameters& sourceParameters, int sourceLocationIndex) const;

    void copyLocationImpl(
        SimulationParameters& targetParameters,
        int targetLocationIndex,
        SimulationParameters& sourceParameters,
        int sourceLocationIndex,
        std::vector<ParameterSpec> const& parameterSpecs) const;
};
