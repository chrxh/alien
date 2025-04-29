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
    void insertDefaultZone(SimulationParameters& parameters, int locationIndex) const; // Create location has locationIndex + 1
    void cloneLocation(SimulationParameters& parameters, int locationIndex) const;  // Create location has locationIndex + 1
    void copyLocation(SimulationParameters& targetParameters, int targetLocationIndex, SimulationParameters& sourceParameters, int sourceLocationIndex) const;

    RadiationStrengths getRadiationStrengths(SimulationParameters const& parameters) const;
    void applyRadiationStrengths(SimulationParameters& parameters, RadiationStrengths const& strengths);

    void adaptRadiationStrengths(RadiationStrengths& strengths, RadiationStrengths& origStrengths, int changeIndex) const;
    RadiationStrengths calcRadiationStrengthsForAddingZone(RadiationStrengths const& strengths) const;
    RadiationStrengths calcRadiationStrengthsForDeletingZone(RadiationStrengths const& strengths, int deleteIndex) const;

private:
    void copyLocationImpl(
        SimulationParameters& targetParameters,
        int targetLocationIndex,
        SimulationParameters& sourceParameters,
        int sourceLocationIndex,
        std::vector<ParameterSpec> const& parameterSpecs) const;
};
