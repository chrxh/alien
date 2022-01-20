#pragma once

#include "Base/Definitions.h"
#include "Descriptions.h"

class DescriptionHelper
{
public:
    ENGINEINTERFACE_EXPORT static void duplicate(ClusteredDataDescription& data, IntVector2D const& origWorldSize, IntVector2D const& worldSize);

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
    ENGINEINTERFACE_EXPORT static DataDescription gridMultiply(DataDescription const& input, GridMultiplyParameters const& parameters);

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
    ENGINEINTERFACE_EXPORT static DataDescription
    randomMultiply(DataDescription const& input, RandomMultiplyParameters const& parameters, IntVector2D const& worldSize);

    ENGINEINTERFACE_EXPORT static void reconnectCells(DataDescription& data, float maxdistance);
    ENGINEINTERFACE_EXPORT static void removeStickiness(DataDescription& data);
    ENGINEINTERFACE_EXPORT static void correctConnections(ClusteredDataDescription& data, IntVector2D const& worldSize);

    ENGINEINTERFACE_EXPORT static void colorize(ClusteredDataDescription& data, std::vector<int> const& colorCodes);

    ENGINEINTERFACE_EXPORT static uint64_t getId(CellOrParticleDescription const& entity);
    ENGINEINTERFACE_EXPORT static RealVector2D getPos(CellOrParticleDescription const& entity);
    ENGINEINTERFACE_EXPORT static std::vector<CellOrParticleDescription> getEntities(DataDescription const& data);

private:
    static void makeValid(DataDescription& data);
    static void makeValid(ClusterDescription& cluster);
};
