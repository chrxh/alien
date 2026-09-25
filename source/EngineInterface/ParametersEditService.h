#pragma once

#include <optional>
#include <set>
#include <string>
#include <vector>

#include <Base/Singleton.h>

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
    static int generateLocationId(SimulationParameters const& parameters);

    void insertDefaultLayer(SimulationParameters& parameters, int orderNumber, int locationId) const;   // Create location at orderNumber + 1
    void insertDefaultSource(SimulationParameters& parameters, int orderNumber, int locationId) const;  // Create location at orderNumber + 1
    void initNewLayer(
        SimulationParameters& parameters,
        int orderNumber,
        IntVector2D const& worldSize,
        RealVector2D const& position,
        FloatColorRGB const& backgroundColor) const;
    void cloneLocation(SimulationParameters& parameters, int orderNumber, int locationId) const;  // Create location at orderNumber + 1
    void deleteLocation(SimulationParameters& parameters, int orderNumber) const;
    void moveLocationUpwards(SimulationParameters& parameters, int orderNumber) const;
    void moveLocationDownwards(SimulationParameters& parameters, int orderNumber) const;

    std::optional<int> insertDefaultLayer(int orderNumber);
    std::optional<int> insertDefaultSource(int orderNumber);
    std::optional<int> cloneLocation(int orderNumber);
    void deleteLocation(int orderNumber);
    void moveLocationUpwards(int orderNumber);
    void moveLocationDownwards(int orderNumber);

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

    RealVector2D calcPositionForNewLocation(IntVector2D const& worldSize) const;

    static inline int _insertedLocationCounter = 0;
    static inline int _lastLocationId = 0;
};
