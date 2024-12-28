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
        MEMBER_DECLARATION(CreateRectParameters, int, width, 10);
        MEMBER_DECLARATION(CreateRectParameters, int, height, 10);
        MEMBER_DECLARATION(CreateRectParameters, float, cellDistance, 1.0f);
        MEMBER_DECLARATION(CreateRectParameters, float, energy, 100.0f);
        MEMBER_DECLARATION(CreateRectParameters, float, stiffness, 1.0f);
        MEMBER_DECLARATION(CreateRectParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER_DECLARATION(CreateRectParameters, bool, removeStickiness, false);
        MEMBER_DECLARATION(CreateRectParameters, int, maxConnections, 6);
        MEMBER_DECLARATION(CreateRectParameters, int, color, 0);
        MEMBER_DECLARATION(CreateRectParameters, bool, barrier, false);
        MEMBER_DECLARATION(CreateRectParameters, bool, randomCreatureId, true);
        MEMBER_DECLARATION(CreateRectParameters, int, mutationId, 0);
        MEMBER_DECLARATION(CreateRectParameters, float, genomeComplexity, 0);
    };
    DataDescription createRect(CreateRectParameters const& parameters);

    struct CreateHexParameters
    {
        MEMBER_DECLARATION(CreateHexParameters, int, layers, 10);
        MEMBER_DECLARATION(CreateHexParameters, float, cellDistance, 1.0f);
        MEMBER_DECLARATION(CreateHexParameters, float, energy, 100.0f);
        MEMBER_DECLARATION(CreateHexParameters, float, stiffness, 1.0f);
        MEMBER_DECLARATION(CreateHexParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER_DECLARATION(CreateHexParameters, bool, removeStickiness, false);
        MEMBER_DECLARATION(CreateHexParameters, int, maxConnections, 6);
        MEMBER_DECLARATION(CreateHexParameters, int, color, 0);
        MEMBER_DECLARATION(CreateHexParameters, bool, barrier, false);
        MEMBER_DECLARATION(CreateHexParameters, bool, randomCreatureId, true);
    };
    DataDescription createHex(CreateHexParameters const& parameters);

    struct CreateUnconnectedCircleParameters
    {
        MEMBER_DECLARATION(CreateUnconnectedCircleParameters, float, radius, 3.0f);
        MEMBER_DECLARATION(CreateUnconnectedCircleParameters, float, cellDistance, 1.0f);
        MEMBER_DECLARATION(CreateUnconnectedCircleParameters, float, energy, 100.0f);
        MEMBER_DECLARATION(CreateUnconnectedCircleParameters, float, stiffness, 1.0f);
        MEMBER_DECLARATION(CreateUnconnectedCircleParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER_DECLARATION(CreateUnconnectedCircleParameters, int, maxConnections, 6);
        MEMBER_DECLARATION(CreateUnconnectedCircleParameters, int, color, 0);
        MEMBER_DECLARATION(CreateUnconnectedCircleParameters, bool, barrier, false);
        MEMBER_DECLARATION(CreateUnconnectedCircleParameters, bool, randomCreatureId, true);
    };
    DataDescription createUnconnectedCircle(CreateUnconnectedCircleParameters const& parameters);

    void duplicate(ClusteredDataDescription& data, IntVector2D const& origWorldSize, IntVector2D const& worldSize);

    struct GridMultiplyParameters
    {
        MEMBER_DECLARATION(GridMultiplyParameters, int, horizontalNumber, 10);
        MEMBER_DECLARATION(GridMultiplyParameters, float, horizontalDistance, 50.0f);
        MEMBER_DECLARATION(GridMultiplyParameters, float, horizontalAngleInc, 0);
        MEMBER_DECLARATION(GridMultiplyParameters, float, horizontalVelXinc, 0);
        MEMBER_DECLARATION(GridMultiplyParameters, float, horizontalVelYinc, 0);
        MEMBER_DECLARATION(GridMultiplyParameters, float, horizontalAngularVelInc, 0);
        MEMBER_DECLARATION(GridMultiplyParameters, int, verticalNumber, 10);
        MEMBER_DECLARATION(GridMultiplyParameters, float, verticalDistance, 50.0f);
        MEMBER_DECLARATION(GridMultiplyParameters, float, verticalAngleInc, 0);
        MEMBER_DECLARATION(GridMultiplyParameters, float, verticalVelXinc, 0);
        MEMBER_DECLARATION(GridMultiplyParameters, float, verticalVelYinc, 0);
        MEMBER_DECLARATION(GridMultiplyParameters, float, verticalAngularVelInc, 0);
    };
    DataDescription gridMultiply(DataDescription const& input, GridMultiplyParameters const& parameters);

    struct RandomMultiplyParameters
    {
        MEMBER_DECLARATION(RandomMultiplyParameters, int, number, 100);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, minAngle, 0);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, maxAngle, 360.0f);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, minVelX, 0);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, maxVelX, 0);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, minVelY, 0);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, maxVelY, 0);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, minAngularVel, 0);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, maxAngularVel, 0);
        MEMBER_DECLARATION(RandomMultiplyParameters, bool, overlappingCheck, false);
    };
    DataDescription randomMultiply(
        DataDescription const& input,
        RandomMultiplyParameters const& parameters,
        IntVector2D const& worldSize,
        DataDescription&& existentData,
        bool& overlappingCheckSuccessful);

    using Occupancy = std::unordered_map<IntVector2D, std::vector<RealVector2D>>;
    void
    addIfSpaceAvailable(DataDescription& result, Occupancy& cellOccupancy, DataDescription const& toAdd, float distance, IntVector2D const& worldSize);

    void reconnectCells(DataDescription& data, float maxDistance);
    void removeStickiness(DataDescription& data);
    void correctConnections(ClusteredDataDescription& data, IntVector2D const& worldSize);

    void randomizeCellColors(ClusteredDataDescription& data, std::vector<int> const& colorCodes);
    void randomizeGenomeColors(ClusteredDataDescription& data, std::vector<int> const& colorCodes);
    void randomizeEnergies(ClusteredDataDescription& data, float minEnergy, float maxEnergy);
    void randomizeAges(ClusteredDataDescription& data, int minAge, int maxAge);
    void randomizeCountdowns(ClusteredDataDescription& data, int minValue, int maxValue);
    void randomizeMutationIds(ClusteredDataDescription& data);

    uint64_t getId(CellOrParticleDescription const& entity);
    RealVector2D getPos(CellOrParticleDescription const& entity);
    std::vector<CellOrParticleDescription> getObjects(DataDescription const& data);
    std::vector<CellOrParticleDescription> getConstructorToMainGenomes(DataDescription const& data);

    void removeMetadata(DataDescription& data);
    void generateNewCreatureIds(DataDescription& data);
    void generateNewCreatureIds(ClusteredDataDescription& data);

private:
    void removeMetadata(CellDescription& cell);
    bool isCellPresent(
        Occupancy const& cellPosBySlot,
        SpaceCalculator const& spaceCalculator,
        RealVector2D const& posToCheck,
        float distance);
};
