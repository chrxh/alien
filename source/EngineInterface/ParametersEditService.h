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

using NewByOldOrderNumber = std::map<int, int>;

class ParametersEditService
{
    MAKE_SINGLETON(ParametersEditService);

public:
    NewByOldOrderNumber insertDefaultLayer(SimulationParameters& parameters, int orderNumber) const;  // Create location at orderNumber + 1
    NewByOldOrderNumber insertDefaultSource(SimulationParameters& parameters, int orderNumber) const;  // Create location at orderNumber + 1
    NewByOldOrderNumber cloneLocation(SimulationParameters& parameters, int orderNumber) const;        // Create location at orderNumber + 1
    NewByOldOrderNumber deleteLocation(SimulationParameters& parameters, int orderNumber) const;
    NewByOldOrderNumber moveLocationUpwards(SimulationParameters& parameters, int orderNumber) const;
    NewByOldOrderNumber moveLocationDownwards(SimulationParameters& parameters, int orderNumber) const;

    RadiationStrengths getRadiationStrengths(SimulationParameters const& parameters) const;
    void applyRadiationStrengths(SimulationParameters& parameters, RadiationStrengths const& strengths);

    void adaptRadiationStrengths(RadiationStrengths& strengths, RadiationStrengths& origStrengths, int changeIndex) const;
    RadiationStrengths calcRadiationStrengthsForAddingSource(RadiationStrengths const& strengths) const;
    RadiationStrengths calcRadiationStrengthsForDeletingLayer(RadiationStrengths const& strengths, int deleteIndex) const;

private:
    void copyLocation(SimulationParameters& targetParameters, int targetOrderNumber, SimulationParameters& sourceParameters, int sourceOrderNumber) const;

    void copyLocationIntern(
        SimulationParameters& targetParameters,
        int targetOrderNumber,
        SimulationParameters& sourceParameters,
        int sourceOrderNumber,
        std::vector<ParameterSpec> const& parameterSpecs) const;
};
