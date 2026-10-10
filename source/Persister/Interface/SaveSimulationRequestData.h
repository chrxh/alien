#pragma once

#include <string>

#include <Base/Interface/MathTypes.h>

struct SaveSimulationRequestData
{
    std::filesystem::path filename;
    float zoom = 1.0f;
    RealVector2D center;
    bool generateNameFromTimestep = false;
};
