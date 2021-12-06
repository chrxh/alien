#pragma once

#include "Base/Definitions.h"
#include "Descriptions.h"

class DescriptionHelper
{
public:
    ENGINEINTERFACE_EXPORT static void
    duplicate(DataDescription2& data, IntVector2D const& origWorldSize, IntVector2D const& worldSize);

    ENGINEINTERFACE_EXPORT static void
    correctConnections(DataDescription2& data, IntVector2D const& worldSize);

    ENGINEINTERFACE_EXPORT static void colorize(DataDescription2& data, std::vector<int> const& colorCodes);

private:
    static void makeValid(ClusterDescription2& cluster);
};
