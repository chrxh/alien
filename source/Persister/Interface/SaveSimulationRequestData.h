#pragma once

#include <filesystem>
#include <string>

#include <Base/MathTypes.h>

#include "SimulationLibrary.h"

struct SaveSimulationRequestData
{
    std::filesystem::path filename;
    float zoom = 1.0f;
    RealVector2D center;
    bool generateNameFromTimestep = false;  // The file is named by the prefix and the time step, in which case filename specifies its folder
    std::string generatedNamePrefix = Const::SavePointPrefix;
};
