#pragma once

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include "Description.h"

class DescriptionEditService
{
    MAKE_SINGLETON(DescriptionEditService);

public:
    struct CreateRectParameters
    {
        MEMBER(CreateRectParameters, int, width, 10);
        MEMBER(CreateRectParameters, int, height, 10);
        MEMBER(CreateRectParameters, CellTypeDescription, cellType, StructureDescription());
        MEMBER(CreateRectParameters, float, cellDistance, 1.0f);
        MEMBER(CreateRectParameters, float, usableEnergy, 100.0f);
        MEMBER(CreateRectParameters, float, rawEnergy, 0.0f);
        MEMBER(CreateRectParameters, float, stiffness, 1.0f);
        MEMBER(CreateRectParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER(CreateRectParameters, bool, sticky, false);
        MEMBER(CreateRectParameters, int, color, 0);
        MEMBER(CreateRectParameters, bool, fixed, false);
    };
    Description createRect(CreateRectParameters const& parameters) const;

    struct CreateHexParameters
    {
        MEMBER(CreateHexParameters, int, layers, 10);
        MEMBER(CreateHexParameters, CellTypeDescription, cellType, StructureDescription());
        MEMBER(CreateHexParameters, float, cellDistance, 1.0f);
        MEMBER(CreateHexParameters, float, usableEnergy, 100.0f);
        MEMBER(CreateHexParameters, float, stiffness, 1.0f);
        MEMBER(CreateHexParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER(CreateHexParameters, bool, sticky, false);
        MEMBER(CreateHexParameters, int, color, 0);
        MEMBER(CreateHexParameters, bool, fixed, false);
    };
    Description createHex(CreateHexParameters const& parameters) const;

    struct CreateUnconnectedCircleParameters
    {
        MEMBER(CreateUnconnectedCircleParameters, float, radius, 3.0f);
        MEMBER(CreateUnconnectedCircleParameters, float, cellDistance, 1.0f);
        MEMBER(CreateUnconnectedCircleParameters, float, usableEnergy, 100.0f);
        MEMBER(CreateUnconnectedCircleParameters, float, stiffness, 1.0f);
        MEMBER(CreateUnconnectedCircleParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER(CreateUnconnectedCircleParameters, int, color, 0);
        MEMBER(CreateUnconnectedCircleParameters, bool, fixed, false);
        MEMBER(CreateUnconnectedCircleParameters, bool, sticky, false);
    };
    Description createUnconnectedCircle(CreateUnconnectedCircleParameters const& parameters) const;

    void duplicate(Description& description, IntVector2D const& origWorldSize, IntVector2D const& worldSize) const;

    struct GridMultiplyParameters
    {
        MEMBER(GridMultiplyParameters, int, horizontalNumber, 10);
        MEMBER(GridMultiplyParameters, float, horizontalDistance, 50.0f);
        MEMBER(GridMultiplyParameters, float, horizontalAngleInc, 0);
        MEMBER(GridMultiplyParameters, float, horizontalVelXinc, 0);
        MEMBER(GridMultiplyParameters, float, horizontalVelYinc, 0);
        MEMBER(GridMultiplyParameters, float, horizontalAngularVelInc, 0);
        MEMBER(GridMultiplyParameters, int, verticalNumber, 10);
        MEMBER(GridMultiplyParameters, float, verticalDistance, 50.0f);
        MEMBER(GridMultiplyParameters, float, verticalAngleInc, 0);
        MEMBER(GridMultiplyParameters, float, verticalVelXinc, 0);
        MEMBER(GridMultiplyParameters, float, verticalVelYinc, 0);
        MEMBER(GridMultiplyParameters, float, verticalAngularVelInc, 0);
    };
    Description gridMultiply(Description const& input, GridMultiplyParameters const& parameters) const;

    struct RandomMultiplyParameters
    {
        MEMBER(RandomMultiplyParameters, int, number, 100);
        MEMBER(RandomMultiplyParameters, float, minAngle, 0);
        MEMBER(RandomMultiplyParameters, float, maxAngle, 360.0f);
        MEMBER(RandomMultiplyParameters, float, minVelX, 0);
        MEMBER(RandomMultiplyParameters, float, maxVelX, 0);
        MEMBER(RandomMultiplyParameters, float, minVelY, 0);
        MEMBER(RandomMultiplyParameters, float, maxVelY, 0);
        MEMBER(RandomMultiplyParameters, float, minAngularVel, 0);
        MEMBER(RandomMultiplyParameters, float, maxAngularVel, 0);
        MEMBER(RandomMultiplyParameters, bool, overlappingCheck, false);
    };
    Description randomMultiply(
        Description const& input,
        RandomMultiplyParameters const& parameters,
        IntVector2D const& worldSize,
        Description&& existentData,
        bool& overlappingCheckSuccessful) const;

    using Occupancy = std::unordered_map<IntVector2D, std::vector<RealVector2D>>;
    void addIfSpaceAvailable(Description& result, Occupancy& cellOccupancy, Description const& toAdd, float distance, IntVector2D const& worldSize) const;

    void flattenTopology(Description& description, IntVector2D const& worldSize) const;

    void reconnectCells(Description& description, float maxDistance) const;  // For non-creatures

    void randomizeCellColors(Description& description, std::vector<int> const& colorCodes) const;
    void randomizeGenomeColors(Description& description, std::vector<int> const& colorCodes) const;
    void randomizeEnergies(Description& description, float minEnergy, float maxEnergy) const;
    void randomizeAges(Description& description, int minAge, int maxAge) const;
    void randomizeCountdowns(Description& description, int minValue, int maxValue) const;
    void randomizeLineageIds(Description& description) const;

    uint64_t getId(ExtendedObjectOrEnergyDescription const& entity) const;
    RealVector2D getPos(ExtendedObjectOrEnergyDescription const& entity) const;
    std::vector<ExtendedObjectOrEnergyDescription> getObjects(Description const& description) const;
    std::vector<ExtendedObjectOrEnergyDescription> getCellsForCreatureRepresentatives(Description const& description) const;

    void setCenter(Description& collection, RealVector2D const& center) const;
    RealVector2D calcCenter(Description const& collection) const;
    void shift(Description& collection, RealVector2D const& delta) const;
    void rotate(Description& collection, float angle) const;
    void accelerate(Description& collection, RealVector2D const& velDelta, float angularVelDelta) const;

    void removeCell(Description& collection, uint64_t objectId) const;
    void removeCellIf(Description& collection, std::function<bool(ObjectDescription const&)> const& predicate) const;

private:
    bool isCellPresent(Occupancy const& cellPosBySlot, SpaceCalculator const& spaceCalculator, RealVector2D const& posToCheck, float distance) const;
};
