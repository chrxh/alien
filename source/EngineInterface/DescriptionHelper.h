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

private:
    static void makeValid(ClusterDescription& cluster);
};
