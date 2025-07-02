#pragma once

#include "Base/Definitions.h"
#include "Base/Singleton.h"

#include "Descriptions.h"

class DescriptionEditService
{
    MAKE_SINGLETON(DescriptionEditService);

public:
    struct CreateRectParameters
    {
        MEMBER(CreateRectParameters, int, width, 10);
        MEMBER(CreateRectParameters, int, height, 10);
        MEMBER(CreateRectParameters, CellTypeDescription, cellType, StructureCellDescription());
        MEMBER(CreateRectParameters, float, cellDistance, 1.0f);
        MEMBER(CreateRectParameters, float, energy, 100.0f);
        MEMBER(CreateRectParameters, float, stiffness, 1.0f);
        MEMBER(CreateRectParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER(CreateRectParameters, bool, sticky, false);
        MEMBER(CreateRectParameters, int, color, 0);
        MEMBER(CreateRectParameters, bool, barrier, false);
    };
    CollectionDescription createRect(CreateRectParameters const& parameters);

    struct CreateHexParameters
    {
        MEMBER(CreateHexParameters, int, layers, 10);
        MEMBER(CreateHexParameters, float, cellDistance, 1.0f);
        MEMBER(CreateHexParameters, float, energy, 100.0f);
        MEMBER(CreateHexParameters, float, stiffness, 1.0f);
        MEMBER(CreateHexParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER(CreateHexParameters, bool, sticky, false);
        MEMBER(CreateHexParameters, int, color, 0);
        MEMBER(CreateHexParameters, bool, barrier, false);
    };
    CollectionDescription createHex(CreateHexParameters const& parameters);

    struct CreateUnconnectedCircleParameters
    {
        MEMBER(CreateUnconnectedCircleParameters, float, radius, 3.0f);
        MEMBER(CreateUnconnectedCircleParameters, float, cellDistance, 1.0f);
        MEMBER(CreateUnconnectedCircleParameters, float, energy, 100.0f);
        MEMBER(CreateUnconnectedCircleParameters, float, stiffness, 1.0f);
        MEMBER(CreateUnconnectedCircleParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER(CreateUnconnectedCircleParameters, int, color, 0);
        MEMBER(CreateUnconnectedCircleParameters, bool, barrier, false);
        MEMBER(CreateUnconnectedCircleParameters, bool, sticky, false);
    };
    CollectionDescription createUnconnectedCircle(CreateUnconnectedCircleParameters const& parameters);

    void duplicate(CollectionDescription& data, IntVector2D const& origWorldSize, IntVector2D const& worldSize);

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
    CollectionDescription gridMultiply(CollectionDescription const& input, GridMultiplyParameters const& parameters);

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
    CollectionDescription randomMultiply(
        CollectionDescription const& input,
        RandomMultiplyParameters const& parameters,
        IntVector2D const& worldSize,
        CollectionDescription&& existentData,
        bool& overlappingCheckSuccessful);

    using Occupancy = std::unordered_map<IntVector2D, std::vector<RealVector2D>>;
    void
    addIfSpaceAvailable(CollectionDescription& result, Occupancy& cellOccupancy, CollectionDescription const& toAdd, float distance, IntVector2D const& worldSize);

    void reconnectCells(CollectionDescription& data, float maxDistance);    // For non-creatures

    void randomizeCellColors(CollectionDescription& data, std::vector<int> const& colorCodes);
    void randomizeGenomeColors(CollectionDescription& data, std::vector<int> const& colorCodes);
    void randomizeEnergies(CollectionDescription& data, float minEnergy, float maxEnergy);
    void randomizeAges(CollectionDescription& data, int minAge, int maxAge);
    void randomizeCountdowns(CollectionDescription& data, int minValue, int maxValue);
    void randomizeMutationIds(CollectionDescription& data);

    uint64_t getId(ExtendedCellOrParticleDescription const& entity);
    RealVector2D getPos(ExtendedCellOrParticleDescription const& entity);
    std::vector<ExtendedCellOrParticleDescription> getObjects(CollectionDescription const& data);
    std::vector<ExtendedCellOrParticleDescription> getCellsForCreatureRepresentatives(CollectionDescription const& data);

    void removeMetadata(CollectionDescription& data);

    void assignNewObjectAndCreatureIds(CollectionDescription& data);

    void setCenter(CollectionDescription& collection, RealVector2D const& center);
    RealVector2D calcCenter(CollectionDescription const& collection) const;
    RealVector2D calcCenter(CreatureDescription const& creature) const;
    void shift(CollectionDescription& collection, RealVector2D const& delta);
    void rotate(CollectionDescription& collection, float angle);
    void accelerate(CollectionDescription& collection, RealVector2D const& velDelta, float angularVelDelta);

private:
    void removeMetadata(CellDescription& cell);
    bool isCellPresent(
        Occupancy const& cellPosBySlot,
        SpaceCalculator const& spaceCalculator,
        RealVector2D const& posToCheck,
        float distance);
};
