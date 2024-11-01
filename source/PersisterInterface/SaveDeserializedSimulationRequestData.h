#pragma once

#include <filesystem>
#include <string>

#include "Base/Vector2D.h"

#include "SharedDeserializedSimulation.h"

struct SaveDeserializedSimulationRequestData
{
    std::filesystem::path filename;
    float zoom = 1.0f;
    RealVector2D center;
    SharedDeserializedSimulation sharedDeserializedSimulation;
};
