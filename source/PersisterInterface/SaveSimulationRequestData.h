#pragma once

#include <string>

#include "Base/Vector2D.h"

struct SaveSimulationRequestData
{
    std::string filename;
    float zoom = 1.0f;
    RealVector2D center;
};
