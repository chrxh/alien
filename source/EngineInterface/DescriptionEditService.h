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
    CollectionDescription createRect(CreateRectParameters const& parameters) const;

    struct CreateHexParameters
    {
        MEMBER(CreateHexParameters, int, layers, 10);
        MEMBER(CreateHexParameters, CellTypeDescription, cellType, StructureCellDescription());
        MEMBER(CreateHexParameters, float, cellDistance, 1.0f);
        MEMBER(CreateHexParameters, float, energy, 100.0f);
        MEMBER(CreateHexParameters, float, stiffness, 1.0f);
        MEMBER(CreateHexParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER(CreateHexParameters, bool, sticky, false);
        MEMBER(CreateHexParameters, int, color, 0);
        MEMBER(CreateHexParameters, bool, barrier, false);
    };
    CollectionDescription createHex(CreateHexParameters const& parameters) const;

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
    CollectionDescription createUnconnectedCircle(CreateUnconnectedCircleParameters const& parameters) const;

    void duplicate(CollectionDescription& data, IntVector2D const& origWorldSize, IntVector2D const& worldSize) const;

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
    CollectionDescription gridMultiply(CollectionDescription const& input, GridMultiplyParameters const& parameters) const;

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
        bool& overlappingCheckSuccessful) const;

    using Occupancy = std::unordered_map<IntVector2D, std::vector<RealVector2D>>;
    void addIfSpaceAvailable(
        CollectionDescription& result,
        Occupancy& cellOccupancy,
        CollectionDescription const& toAdd,
        float distance,
        IntVector2D const& worldSize) const;

    void reconnectCells(CollectionDescription& data, float maxDistance) const;  // For non-creatures

    void randomizeCellColors(CollectionDescription& data, std::vector<int> const& colorCodes) const;
    void randomizeGenomeColors(CollectionDescription& data, std::vector<int> const& colorCodes) const;
    void randomizeEnergies(CollectionDescription& data, float minEnergy, float maxEnergy) const;
    void randomizeAges(CollectionDescription& data, int minAge, int maxAge) const;
    void randomizeCountdowns(CollectionDescription& data, int minValue, int maxValue) const;
    void randomizeMutationIds(CollectionDescription& data) const;

    uint64_t getId(ExtendedCellOrParticleDescription const& entity) const;
    RealVector2D getPos(ExtendedCellOrParticleDescription const& entity) const;
    std::vector<ExtendedCellOrParticleDescription> getObjects(CollectionDescription const& data) const;
    std::vector<ExtendedCellOrParticleDescription> getCellsForCreatureRepresentatives(CollectionDescription const& data) const;

    void assignNewObjectAndCreatureIds(CollectionDescription& data) const;

    void setCenter(CollectionDescription& collection, RealVector2D const& center) const;
    RealVector2D calcCenter(CollectionDescription const& collection) const;
    RealVector2D calcCenter(CreatureDescription const& creature) const;
    void shift(CollectionDescription& collection, RealVector2D const& delta) const;
    void rotate(CollectionDescription& collection, float angle) const;
    void accelerate(CollectionDescription& collection, RealVector2D const& velDelta, float angularVelDelta) const;

    void removeCell(CollectionDescription& collection, uint64_t cellId) const;

private:
    bool isCellPresent(
        Occupancy const& cellPosBySlot,
        SpaceCalculator const& spaceCalculator,
        RealVector2D const& posToCheck,
        float distance) const;
};
