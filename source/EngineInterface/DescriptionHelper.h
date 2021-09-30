#pragma once

#include "Base/Definitions.h"
#include "Descriptions.h"

class DescriptionHelper
{
public:
    ENGINEINTERFACE_EXPORT static void
    duplicate(DataDescription& data, IntVector2D const& origSize, IntVector2D const& size);

private:
    static void makeValid(ClusterDescription& cluster);
};
