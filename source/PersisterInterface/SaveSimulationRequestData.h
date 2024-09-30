#pragma once

#include <string>

#include "Base/Vector2D.h"

struct SaveSimulationRequestData
{
    std::string filename;
    float zoom;
    RealVector2D center;
};
