#pragma once

#include "Base/Definitions.h"
#include "Descriptions.h"

class DescriptionHelper
{
public:
    ENGINEINTERFACE_EXPORT static void
    duplicate(DataDescription& data, IntVector2D const& origWorldSize, IntVector2D const& worldSize);

    ENGINEINTERFACE_EXPORT static void
    correctConnections(DataDescription& data, IntVector2D const& worldSize);

    ENGINEINTERFACE_EXPORT static void colorize(DataDescription& data, std::vector<int> const& colorCodes);

    ENGINEINTERFACE_EXPORT static uint64_t getId(CellOrParticleDescription const& entity);
    ENGINEINTERFACE_EXPORT static RealVector2D getPos(CellOrParticleDescription const& entity);

private:
    static void makeValid(ClusterDescription& cluster);
};
