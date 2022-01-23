#pragma once

#include "Base/Definitions.h"
#include "Descriptions.h"

class DescriptionHelper
{
public:
    struct CreateRectParameters
    {
        MEMBER_DECLARATION(CreateRectParameters, int, width, 10);
        MEMBER_DECLARATION(CreateRectParameters, int, height, 10);
        MEMBER_DECLARATION(CreateRectParameters, float, cellDistance, 1.0f);
        MEMBER_DECLARATION(CreateRectParameters, float, energy, 100.0f);
        MEMBER_DECLARATION(CreateRectParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER_DECLARATION(CreateRectParameters, bool, removeStickiness, false);
        MEMBER_DECLARATION(CreateRectParameters, int, maxConnection, 6);
        MEMBER_DECLARATION(CreateRectParameters, int, color, 0);
    };
    static DataDescription createRect(CreateRectParameters const& parameters);

    static void duplicate(ClusteredDataDescription& data, IntVector2D const& origWorldSize, IntVector2D const& worldSize);

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
    static DataDescription gridMultiply(DataDescription const& input, GridMultiplyParameters const& parameters);

    struct RandomMultiplyParameters
    {
        MEMBER_DECLARATION(RandomMultiplyParameters, int, number, 100);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, minAngle, 0);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, maxAngle, 0);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, minVelX, 0);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, maxVelX, 0);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, minVelY, 0);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, maxVelY, 0);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, minAngularVel, 0);
        MEMBER_DECLARATION(RandomMultiplyParameters, float, maxAngularVel, 0);
    };
    static DataDescription randomMultiply(DataDescription const& input, RandomMultiplyParameters const& parameters, IntVector2D const& worldSize);

    static void reconnectCells(DataDescription& data, float maxdistance);
    static void removeStickiness(DataDescription& data);
    static void correctConnections(ClusteredDataDescription& data, IntVector2D const& worldSize);

    static void colorize(ClusteredDataDescription& data, std::vector<int> const& colorCodes);

    static uint64_t getId(CellOrParticleDescription const& entity);
    static RealVector2D getPos(CellOrParticleDescription const& entity);
    static std::vector<CellOrParticleDescription> getEntities(DataDescription const& data);

private:
    static void makeValid(DataDescription& data);
    static void makeValid(ClusterDescription& cluster);
};
