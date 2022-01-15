#pragma once

#include "Base/Definitions.h"
#include "Descriptions.h"

class DescriptionHelper
{
public:
    ENGINEINTERFACE_EXPORT static void duplicate(ClusteredDataDescription& data, IntVector2D const& origWorldSize, IntVector2D const& worldSize);

    ENGINEINTERFACE_EXPORT static void reconnectCells(std::vector<CellDescription>& cells, float maxdistance);
    ENGINEINTERFACE_EXPORT static void correctConnections(ClusteredDataDescription& data, IntVector2D const& worldSize);

    ENGINEINTERFACE_EXPORT static void colorize(ClusteredDataDescription& data, std::vector<int> const& colorCodes);

    ENGINEINTERFACE_EXPORT static uint64_t getId(CellOrParticleDescription const& entity);
    ENGINEINTERFACE_EXPORT static RealVector2D getPos(CellOrParticleDescription const& entity);
    ENGINEINTERFACE_EXPORT static std::vector<CellOrParticleDescription> getEntities(DataDescription const& data);

private:
    static void makeValid(ClusterDescription& cluster);
};
