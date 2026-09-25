#pragma once

#include <optional>
#include <string>
#include <vector>

#include <Base/Singleton.h>

#include "SimulationParameters.h"
#include "SimulationParametersSpecification.h"

class LocationEditService
{
    MAKE_SINGLETON(LocationEditService);

public:
    std::optional<int> insertDefaultLayer(SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber, IntVector2D const& worldSize)
        const;
    std::optional<int>
    insertDefaultSource(SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber, IntVector2D const& worldSize) const;
    std::optional<int> cloneLocation(SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber) const;
    void deleteLocation(SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber) const;
    void moveLocationUpwards(SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber) const;
    void moveLocationDownwards(SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber) const;

    int generateLocationId(SimulationParameters const& parameters) const;
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

    std::string generateLayerName(SimulationParameters const& parameters) const;
    std::string generateSourceName(SimulationParameters const& parameters) const;
    void assignLocationIds(SimulationParameters& parameters) const;

private:
    int& findOrderNumberRef(SimulationParameters& parameters, int orderNumber) const;
    void decreaseOrderNumber(SimulationParameters& parameters, int orderNumber) const;
    void increaseOrderNumber(SimulationParameters& parameters, int orderNumber) const;
    void adaptLocationIndices(SimulationParameters& parameters, int fromOrderNumber, int offset) const;
    void setLocationId(SimulationParameters& parameters, int orderNumber, int locationId) const;

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
