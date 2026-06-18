#pragma once

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include "Desc.h"

class DescEditService
{
    MAKE_SINGLETON(DescEditService);

public:
    struct CreateRectParameters
    {
        MEMBER(CreateRectParameters, int, width, 10);
        MEMBER(CreateRectParameters, int, height, 10);
        MEMBER(CreateRectParameters, ObjectTypeDesc, objectType, SolidDesc());
        MEMBER(CreateRectParameters, float, cellDistance, 1.0f);
        MEMBER(CreateRectParameters, bool, connectObjects, true);
        MEMBER(CreateRectParameters, float, stiffness, 1.0f);
        MEMBER(CreateRectParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER(CreateRectParameters, bool, sticky, false);
        MEMBER(CreateRectParameters, int, color, 0);
        MEMBER(CreateRectParameters, bool, isStatic, false);
    };
    Desc createRect(CreateRectParameters const& parameters) const;

    struct CreateHexParameters
    {
        MEMBER(CreateHexParameters, int, layers, 10);
        MEMBER(CreateHexParameters, ObjectTypeDesc, objectType, SolidDesc());
        MEMBER(CreateHexParameters, float, cellDistance, 1.0f);
        MEMBER(CreateHexParameters, bool, connectObjects, true);
        MEMBER(CreateHexParameters, float, stiffness, 1.0f);
        MEMBER(CreateHexParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER(CreateHexParameters, bool, sticky, false);
        MEMBER(CreateHexParameters, int, color, 0);
        MEMBER(CreateHexParameters, bool, isStatic, false);
    };
    Desc createHex(CreateHexParameters const& parameters) const;

    struct CreateCircleParameters
    {
        MEMBER(CreateCircleParameters, ObjectTypeDesc, type, SolidDesc());
        MEMBER(CreateCircleParameters, float, radius, 3.0f);
        MEMBER(CreateCircleParameters, float, cellDistance, 1.0f);
        MEMBER(CreateCircleParameters, bool, connectObjects, true);
        MEMBER(CreateCircleParameters, float, stiffness, 1.0f);
        MEMBER(CreateCircleParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER(CreateCircleParameters, int, color, 0);
        MEMBER(CreateCircleParameters, bool, isStatic, false);
        MEMBER(CreateCircleParameters, bool, sticky, false);
    };
    Desc createCircle(CreateCircleParameters const& parameters) const;

    void duplicate(Desc& description, IntVector2D const& origWorldSize, IntVector2D const& worldSize) const;

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
    Desc gridMultiply(Desc const& input, GridMultiplyParameters const& parameters) const;

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
    Desc randomMultiply(
        Desc const& input,
        RandomMultiplyParameters const& parameters,
        IntVector2D const& worldSize,
        Desc&& existentData,
        bool& overlappingCheckSuccessful) const;

    using Occupancy = std::unordered_map<IntVector2D, std::vector<RealVector2D>>;
    void addIfSpaceAvailable(Desc& result, Occupancy& occupancy, Desc const& toAdd, float distance, IntVector2D const& worldSize) const;

    void flattenTopology(Desc& description, IntVector2D const& worldSize) const;

    void reconnectObjects(Desc& description, float maxDistance) const;  // For non-creatures

    void randomizeCellColors(Desc& description, std::vector<int> const& colorCodes) const;
    void randomizeGenomeColors(Desc& description, std::vector<int> const& colorCodes) const;
    void randomizeEnergies(Desc& description, float minEnergy, float maxEnergy) const;
    void randomizeAges(Desc& description, int minAge, int maxAge) const;
    void randomizeCountdowns(Desc& description, int minValue, int maxValue) const;
    void randomizeLineageIds(Desc& description) const;
    void randomizeGlow(Desc& description, float minGlow, float maxGlow) const;
    void setMutationRates(Desc& description, MutationRatesDesc const& mutationRates) const;

    uint64_t getId(ExtendedObjectOrEnergyDesc const& entity) const;
    RealVector2D getPos(ExtendedObjectOrEnergyDesc const& entity) const;
    std::vector<ExtendedObjectOrEnergyDesc> getObjects(Desc const& description) const;

    void setCenter(Desc& collection, RealVector2D const& center) const;
    RealVector2D calcCenter(Desc const& collection) const;
    void shift(Desc& collection, RealVector2D const& delta) const;
    void rotate(Desc& collection, float angle) const;
    void accelerate(Desc& collection, RealVector2D const& velDelta, float angularVelDelta) const;

    void removeCell(Desc& collection, uint64_t objectId) const;
    void removeCellIf(Desc& collection, std::function<bool(ObjectDesc const&)> const& predicate) const;

private:
    std::vector<std::vector<size_t>> calcClusters(Desc const& description) const;

    bool isCellPresent(Occupancy const& cellPosBySlot, SpaceCalculator const& spaceCalculator, RealVector2D const& posToCheck, float distance) const;
};
